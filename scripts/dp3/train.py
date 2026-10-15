#!/usr/bin/env python3
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a DP3 policy on DexVerse demonstrations.

Usage:
  python scripts/dp3/train.py \\
      --hdf5_path outputs/dp3_graspPot.hdf5 \\
      --output_dir runs/dp3_graspPot \\
      --num_epochs 3000 \\
      --batch_size 128
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(int(item) for item in items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DP3 on DexVerse demonstrations.")

    # Dataset
    parser.add_argument("--hdf5_path", type=str, required=True, help="DP3 HDF5 dataset path")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=8)

    # Training
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", action="store_true", default=False)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_every_epochs", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=5)

    # Policy architecture
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--num_train_timesteps", type=int, default=100)
    parser.add_argument("--encoder_output_dim", type=int, default=64)
    parser.add_argument("--down_dims", type=str, default="512,1024,2048")
    parser.add_argument("--condition_type", type=str, default="film")
    parser.add_argument("--diffusion_step_embed_dim", type=int, default=128)

    return parser.parse_args()


def main() -> None:
    from dexverse.IL.dp3.config import (
        DP3DatasetConfig,
        DP3PolicyConfig,
        DP3TrainConfig,
        PointcloudEncoderConfig,
    )
    from dexverse.IL.dp3.train import train_main

    args = parse_args()

    dataset_cfg = DP3DatasetConfig(
        hdf5_path=args.hdf5_path,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )

    policy_cfg = DP3PolicyConfig(
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        num_inference_steps=args.num_inference_steps,
        num_train_timesteps=args.num_train_timesteps,
        encoder_output_dim=args.encoder_output_dim,
        down_dims=_parse_int_tuple(args.down_dims),
        condition_type=args.condition_type,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        pointcloud_encoder_cfg=PointcloudEncoderConfig(),
    )

    train_cfg = DP3TrainConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_ema=args.use_ema and not args.no_ema,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        save_every_epochs=args.save_every_epochs,
        val_every=args.val_every,
        log_every=args.log_every,
    )

    train_main(train_cfg)


if __name__ == "__main__":
    main()
