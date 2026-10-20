#!/usr/bin/env python3
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the state-based Diffusion Policy baseline on a demonstration dataset.

Takes the HDF5 written by ``build_dataset.py``. Pure PyTorch, so pick a GPU with
``CUDA_VISIBLE_DEVICES``. Launch with ``torchrun --nproc_per_node N`` for multi-GPU
data parallel training (``--batch_size`` stays the global batch).

Usage::

    python scripts/diffusion/train.py \
        --task Dexverse-GraspKettle-v0 \
        --dataset_file datasets/diffusion/Dexverse-GraspKettle-v0.h5 \
        --output_dir runs/dp_Dexverse-GraspKettle-v0
"""

from __future__ import annotations

import argparse

from dexverse.IL.diffusion.config import TrainConfig
from dexverse.IL.diffusion.train import train


def main() -> None:
    defaults = TrainConfig(task="", dataset_files=(), output_dir="")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, help="Task id the demonstrations were recorded on.")
    parser.add_argument("--dataset_file", action="append", dest="dataset_files", required=True,
                        help="Training HDF5 from build_dataset.py. Repeatable.")
    parser.add_argument("--output_dir", required=True, help="Where checkpoints and metrics.json go.")
    parser.add_argument("--obs_horizon", type=int, default=defaults.obs_horizon)
    parser.add_argument("--action_horizon", type=int, default=defaults.action_horizon)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--ema_decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--save_every_epochs", type=int, default=defaults.save_every_epochs)
    parser.add_argument("--split_ratio", type=float, default=defaults.split_ratio,
                        help="Fraction of episodes for training (per-episode split); the rest validate.")
    parser.add_argument("--slim_checkpoints", action="store_true",
                        help="Save only EMA weights in last.pt/best.pt (half the size; no training resume).")
    parser.add_argument("--no_val", action="store_true",
                        help="Use every episode for training (no validation split, early stopping or best.pt).")
    parser.add_argument("--no_save_last", dest="save_last", action="store_false",
                        help="Do not write last.pt; with best.pt enabled, only the best-validation checkpoint is kept.")
    parser.add_argument("--no_save_best", dest="save_best", action="store_false",
                        help="Do not write best.pt; keep only last.pt (plus epoch_*.pt if enabled).")
    parser.add_argument("--early_stopping_patience", type=int, default=defaults.early_stopping_patience,
                        help="Epochs without a new best validation loss before stopping; 0 disables.")
    parser.add_argument("--epoch_dir", default=defaults.epoch_dir, help="Folder for epoch_*.pt (default: --output_dir).")
    parser.add_argument("--grad_clip_norm", type=float, default=defaults.grad_clip_norm, help="0 disables clipping.")
    parser.add_argument("--unet_type", choices=("diffusers", "chi"), default=defaults.unet_type,
                        help="chi = official Diffusion Policy ConditionalUnet1D with FiLM global conditioning.")
    parser.add_argument("--unet_block_out_channels", type=int, nargs="+", default=list(defaults.unet_block_out_channels))
    parser.add_argument("--diffusion_step_embed_dim", type=int, default=defaults.diffusion_step_embed_dim)
    parser.add_argument("--kernel_size", type=int, default=defaults.kernel_size)
    parser.add_argument("--n_groups", type=int, default=defaults.n_groups)
    parser.add_argument("--no_cond_predict_scale", dest="cond_predict_scale", action="store_false")
    parser.add_argument("--state_groups", nargs="+", default=list(defaults.state_groups),
                        help="Observation groups forming the policy state (latest step of history-stacked groups).")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=list(defaults.adam_betas))
    parser.add_argument("--lr_scheduler", choices=("constant", "cosine"), default=defaults.lr_scheduler)
    parser.add_argument("--lr_warmup_steps", type=int, default=defaults.lr_warmup_steps)
    parser.add_argument("--ema_power", type=float, default=defaults.ema_power,
                        help="> 0: official DP EMA warmup schedule (decay capped at --ema_max_decay); 0: fixed --ema_decay.")
    parser.add_argument("--ema_inv_gamma", type=float, default=defaults.ema_inv_gamma)
    parser.add_argument("--ema_max_decay", type=float, default=defaults.ema_max_decay)
    parser.add_argument("--image_crop", type=int, default=defaults.image_crop,
                        help="Camera encoder crop (random at train, center at eval); 0 = 7/8 of the stored image size.")
    parser.add_argument("--image_feature_dim", type=int, default=defaults.image_feature_dim)
    parser.add_argument("--num_keypoints", type=int, default=defaults.num_keypoints)
    parser.add_argument("--resume_file", default=defaults.resume_file,
                        help="Full training state, rewritten every --resume_every_epochs; training continues from it "
                             "if it exists, and it is deleted when training finishes.")
    parser.add_argument("--resume_every_epochs", type=int, default=defaults.resume_every_epochs)
    args = parser.parse_args()

    cfg = TrainConfig(**{**vars(args), "dataset_files": tuple(args.dataset_files),
                         "unet_block_out_channels": tuple(args.unet_block_out_channels),
                         "state_groups": tuple(args.state_groups), "adam_betas": tuple(args.adam_betas)})
    metrics = train(cfg)
    if metrics is not None:
        print(metrics)


if __name__ == "__main__":
    main()
