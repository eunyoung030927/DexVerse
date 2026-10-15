# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for the DP3 integration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PointcloudEncoderConfig:
    """PointNet encoder configuration."""

    in_channels: int = 3
    out_channels: int = 64
    use_layernorm: bool = True
    final_norm: str = "layernorm"
    normal_channel: bool = False


@dataclass
class DP3PolicyConfig:
    """DP3 policy network configuration."""

    # Observation dimensions (derived from dataset at runtime)
    num_points: int = 512
    proprio_dim: int = 0
    action_dim: int = 0

    # Temporal horizons
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8

    # Diffusion
    num_inference_steps: int = 10
    num_train_timesteps: int = 100

    # Architecture
    encoder_output_dim: int = 64
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    condition_type: str = "film"
    use_down_condition: bool = True
    use_mid_condition: bool = True
    use_up_condition: bool = True
    use_pc_color: bool = False
    pointnet_type: str = "pointnet"

    # PointNet encoder
    pointcloud_encoder_cfg: PointcloudEncoderConfig = field(default_factory=PointcloudEncoderConfig)


@dataclass
class DP3DatasetConfig:
    """Configuration for the DP3 dataset loader."""

    hdf5_path: str = ""
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    seed: int = 42
    val_ratio: float = 0.1


@dataclass
class DP3TrainConfig:
    """Training loop configuration."""

    # Dataset
    dataset: DP3DatasetConfig = field(default_factory=DP3DatasetConfig)

    # Policy (dimensions filled from dataset at runtime)
    policy: DP3PolicyConfig = field(default_factory=DP3PolicyConfig)

    # Training
    output_dir: str = "runs/dp3"
    batch_size: int = 128
    num_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-6
    betas: tuple[float, float] = (0.95, 0.999)
    use_ema: bool = True
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 8

    # Logging / checkpointing
    save_every_epochs: int = 10
    val_every: int = 1
    log_every: int = 5


@dataclass
class DP3OnlineEvalConfig:
    """Online evaluation configuration."""

    ckpt: str = ""
    task: str = ""
    output_dir: str = "runs/dp3_eval_online"
    num_episodes: int = 20
    max_steps: int = 500
    device: str = "cuda"
    replan_interval: int = 8
    num_points: int = 512
    json_path: str | None = None
    # Observation preset applied to the env cfg before gym.make(). Must match
    # the preset used when training data was generated. "pointcloud" enables
    # {policy, proprio, goal, pointcloud}; "3view_pointcloud" swaps the
    # single-source point cloud for a 3-view merged cloud.
    observation_preset: str = "pointcloud"

    # Video recording — write one MP4 per episode from a named camera sensor.
    record_video: bool = False
    video_camera: str = "third_person_camera"
    video_fps: int = 30
