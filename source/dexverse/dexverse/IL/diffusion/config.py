# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for the diffusion baseline.

Defaults are the settings the reported baseline numbers were produced with.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Sliding-window view over one or more training HDF5 files."""

    dataset_files: tuple[str, ...]
    obs_horizon: int = 2
    action_horizon: int = 16
    split: str = "train"
    """``train``, ``val`` or ``all``; splits are per-episode, not per-window."""
    split_ratio: float = 0.95
    seed: int = 0
    state_index: tuple[int, ...] = ()
    """Columns of ``obs/state`` to keep (``obs_state.select_state_index``); empty = all."""


@dataclass
class PolicyConfig:
    """Architecture of the conditional 1-D UNet and its noise schedule."""

    state_dim: int
    action_dim: int
    obs_horizon: int
    action_horizon: int
    obs_embed_dim: int = 256
    unet_block_out_channels: tuple[int, ...] = (256, 512, 1024)
    layers_per_block: int = 2
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    image_keys: tuple[str, ...] = ()
    """Camera term names (``rgb`` group); one image encoder per camera. Empty = state-only policy."""
    image_size: int = 0
    """Square resolution the stored / live images are resized to (``obs_images.resize_images``)."""
    image_crop: int = 0
    """Crop fed to the encoders: random during training, centered at inference."""
    image_feature_dim: int = 64
    num_keypoints: int = 32
    """Spatial-softmax keypoints per camera (ResNet18 + GroupNorm + spatial softmax, as in Chi et al.)."""
    unet_type: str = "diffusers"
    """``diffusers``: DexVerse baseline (diffusers UNet1DModel, obs MLP added to the timestep embedding).
    ``chi``: official Diffusion Policy ConditionalUnet1D, observation features as the global FiLM condition."""
    diffusion_step_embed_dim: int = 128
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    state_index: tuple[int, ...] = ()
    """Indices into the full ``obs_state.build_state`` vector that form the policy state; empty = all of it."""
    full_state_dim: int = 0
    """Length of the full ``build_state`` vector ``state_index`` selects from (0 = no selection)."""


@dataclass
class TrainConfig:
    task: str
    dataset_files: tuple[str, ...]
    output_dir: str
    obs_horizon: int = 2
    action_horizon: int = 16
    batch_size: int = 256
    epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.995
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 0
    save_every_epochs: int = 10
    """Also write ``epoch_*.pt`` every N epochs (and at the final / early-stop epoch); 0 disables."""
    epoch_dir: str = ""
    """Where ``epoch_*.pt`` go (default ``output_dir``), e.g. a checkpoints folder for rollout-based selection."""
    split_ratio: float = 0.95
    """Fraction of episodes used for training; the rest are the validation split."""
    slim_checkpoints: bool = False
    """Store only the EMA weights (what evaluation loads) in last.pt/best.pt; halves the file size, but cannot resume."""
    no_val: bool = False
    """Train on every episode (split ``all``): no validation, so no early stopping or best.pt."""
    save_last: bool = True
    """Write ``last.pt`` every epoch."""
    save_best: bool = True
    """Write ``best.pt`` (EMA weights at the best validation loss). ``last.pt`` is always written."""
    early_stopping_patience: int = 40
    """Stop after this many epochs without a new best validation loss; 0 disables."""
    obs_embed_dim: int = 256
    unet_block_out_channels: tuple[int, ...] = (256, 512, 1024)
    layers_per_block: int = 2
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    image_crop: int = 0
    """Image crop for the camera encoders; 0 = 7/8 of the stored image size (used only when the dataset has images)."""
    image_feature_dim: int = 64
    num_keypoints: int = 32
    unet_type: str = "diffusers"
    diffusion_step_embed_dim: int = 128
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    state_groups: tuple[str, ...] = ()
    """Observation groups that form the policy state (e.g. ``proprio goal``); empty = every group in the dataset.
    Groups the env stacks with history (``history`` in the dataset layout) keep only their latest step."""
    adam_betas: tuple[float, float] = (0.9, 0.999)
    lr_scheduler: str = "constant"
    """``constant`` or ``cosine`` (cosine decay over all training steps after ``lr_warmup_steps``)."""
    lr_warmup_steps: int = 0
    ema_power: float = 0.0
    """0: fixed ``ema_decay``. > 0: decay = 1 - (1 + step / ema_inv_gamma) ** -ema_power, capped at ``ema_max_decay``
    (the official Diffusion Policy EMA schedule)."""
    ema_inv_gamma: float = 1.0
    ema_max_decay: float = 0.9999
    resume_file: str = ""
    """Full training state (weights, EMA, optimizer, lr schedule, history), rewritten every ``resume_every_epochs``.
    If the file exists at start, training continues from it; it is deleted when training finishes. Empty = off."""
    resume_every_epochs: int = 5


@dataclass
class EvalConfig:
    task: str
    ckpt: str
    output_dir: str = "runs/diffusion_eval"
    num_episodes: int = 50
    max_steps: int = 1000
    device: str = "cuda:0"
    sample_steps: int = 20
    """DDPM denoising steps per replan. Fewer than ``num_train_timesteps`` is fine."""
    sampler: str = "ddpm"
    """``ddpm`` or ``ddim`` (deterministic, eta 0) denoising at inference; ``sample_steps`` is its step count."""
    replan_interval: int = 4
    """Env steps between replans; the rest of the sampled chunk is played open-loop."""
    observation_preset: str | None = None
    """Preset to narrow the live env's observation groups to. Must match the
    ``--obs-groups`` preset the demos were replayed with, or optional groups
    appear at eval that were absent at training time and ``state_dim`` breaks."""
    reset_from_dataset: str | None = None
    """Training HDF5 to pull per-episode ``initial_state`` from, pinning episode
    starts to recorded ones instead of the env's reset randomization. A
    diagnostic: it separates policy quality from reset distribution shift."""
    video: bool = False
    video_length: int | None = None
    json_path: str | None = None
    """Task-specific object/scene JSON, forwarded to ``parse_env_cfg``."""
    policy_device: str | None = None
    """Device for policy inference (default: ``device``). Physics stays on ``device``, e.g. ``--device cpu
    --policy_device cuda:0`` keeps CPU physics like the recordings while denoising on the GPU."""
    seed: int | None = None
    """Env seed for reset randomization; give parallel evaluation shards different seeds."""
    robot_type: str | None = None
    """Override the task's hand (e.g. ``floating_allegro_right``); must match the demos the policy was trained on."""
