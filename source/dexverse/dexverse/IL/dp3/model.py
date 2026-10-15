# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DP3 policy factory for DexVerse."""
from __future__ import annotations

import copy

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from omegaconf import OmegaConf

from .config import DP3PolicyConfig
from .dp3_core.dp3_policy import DP3
from .dp3_core.ema_model import EMAModel
from .dp3_core.normalizer import LinearNormalizer


def build_dp3_policy(cfg: DP3PolicyConfig) -> DP3:
    """Construct a DP3 policy from a DexVerse config.

    Uses OmegaConf wrappers for ``shape_meta`` and ``pointcloud_encoder_cfg``
    so that DP3's internal ``dict_apply`` does not recurse into them.
    """
    shape_meta = OmegaConf.create({
        "obs": {
            "point_cloud": {"shape": [cfg.num_points, 3], "type": "point_cloud"},
            "agent_pos": {"shape": [cfg.proprio_dim], "type": "low_dim"},
        },
        "action": {"shape": [cfg.action_dim]},
    })

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="sample",
    )

    pcd_enc = OmegaConf.create({
        "in_channels": cfg.pointcloud_encoder_cfg.in_channels,
        "out_channels": cfg.pointcloud_encoder_cfg.out_channels,
        "use_layernorm": cfg.pointcloud_encoder_cfg.use_layernorm,
        "final_norm": cfg.pointcloud_encoder_cfg.final_norm,
        "normal_channel": cfg.pointcloud_encoder_cfg.normal_channel,
    })

    return DP3(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=cfg.horizon,
        n_action_steps=cfg.n_action_steps,
        n_obs_steps=cfg.n_obs_steps,
        num_inference_steps=cfg.num_inference_steps,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        down_dims=cfg.down_dims,
        kernel_size=cfg.kernel_size,
        n_groups=cfg.n_groups,
        condition_type=cfg.condition_type,
        use_down_condition=cfg.use_down_condition,
        use_mid_condition=cfg.use_mid_condition,
        use_up_condition=cfg.use_up_condition,
        encoder_output_dim=cfg.encoder_output_dim,
        use_pc_color=cfg.use_pc_color,
        pointnet_type=cfg.pointnet_type,
        pointcloud_encoder_cfg=pcd_enc,
    )


def build_ema_model(policy: DP3) -> EMAModel:
    """Create an EMA copy of the policy."""
    ema_policy = copy.deepcopy(policy)
    return EMAModel(
        model=ema_policy,
        update_after_step=0,
        inv_gamma=1.0,
        power=0.75,
        min_value=0.0,
        max_value=0.9999,
    )


def load_dp3_checkpoint(
    ckpt_path: str,
    device: str = "cpu",
) -> dict:
    """Load a DP3 training checkpoint.

    Returns dict with keys: policy_state, ema_policy_state, normalizer_state,
    policy_cfg, dataset_cfg, optimizer_state, epoch, metadata.
    """
    import torch

    return torch.load(ckpt_path, map_location=device, weights_only=False)
