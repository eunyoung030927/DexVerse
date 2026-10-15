# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DP3 training loop for DexVerse."""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DP3DatasetConfig, DP3PolicyConfig, DP3TrainConfig
from .dataset import DexVerseDP3Dataset
from .model import build_dp3_policy, build_ema_model


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


@torch.no_grad()
def _evaluate(policy, val_loader, device) -> float:
    """Compute average validation loss."""
    policy.eval()
    losses = []
    for batch in val_loader:
        batch = _batch_to_device(batch, device)
        loss, _ = policy.compute_loss(batch)
        losses.append(loss.item())
    policy.train()
    return float(np.mean(losses)) if losses else 0.0


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """Recursively move batch tensors to device."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            result[k] = _batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        else:
            result[k] = v
    return result


def _save_checkpoint(
    path: str,
    *,
    policy,
    ema_model,
    normalizer,
    optimizer,
    policy_cfg: DP3PolicyConfig,
    dataset_cfg: DP3DatasetConfig,
    epoch: int,
    metadata: dict,
    save_optimizer: bool = True,
) -> None:
    """Save a training checkpoint.

    ``save_optimizer`` controls whether the AdamW state (~2x the model size) is
    written. It is only needed to resume training; inference/eval never reads
    it, so the sweep skips it to keep checkpoints small and I/O cheap.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_state": policy.state_dict(),
        "normalizer_state": normalizer.state_dict(),
        "policy_cfg": dataclasses.asdict(policy_cfg),
        "dataset_cfg": dataclasses.asdict(dataset_cfg),
        "epoch": epoch,
        "metadata": metadata,
    }
    if save_optimizer:
        payload["optimizer_state"] = optimizer.state_dict()
    if ema_model is not None:
        payload["ema_policy_state"] = ema_model.averaged_model.state_dict()
    torch.save(payload, path)


def train_main(cfg: DP3TrainConfig) -> dict[str, Any]:
    """Run the DP3 training loop.

    Returns a summary dict with final metrics.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    _seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)

    # ---- Dataset ----
    train_dataset = DexVerseDP3Dataset(cfg.dataset, split="train")
    val_dataset = train_dataset.get_validation_dataset()

    print(f"[dp3][train] Train: {len(train_dataset)} samples from {len(train_dataset._active_indices)} episodes")
    print(f"[dp3][train] Val:   {len(val_dataset)} samples from {len(val_dataset._active_indices)} episodes")
    print(
        f"[dp3][train] Dims: proprio={train_dataset.proprio_dim}, "
        f"action={train_dataset.action_dim}, points={train_dataset.num_points}"
    )

    # ---- Normalizer ----
    normalizer = train_dataset.get_normalizer()

    # ---- Policy ----
    policy_cfg = dataclasses.replace(
        cfg.policy,
        proprio_dim=train_dataset.proprio_dim,
        action_dim=train_dataset.action_dim,
        num_points=train_dataset.num_points,
    )
    policy = build_dp3_policy(policy_cfg)
    policy.set_normalizer(normalizer)
    policy = policy.to(device)

    # ---- EMA ----
    ema_model = build_ema_model(policy) if cfg.use_ema else None

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) >= cfg.batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ---- Training loop ----
    best_val_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []
    start_time = time.time()

    metadata = {
        "task": train_dataset.metadata.get("task", "unknown"),
        "bbox_min": train_dataset.bbox_min.tolist() if train_dataset.bbox_min is not None else None,
        "bbox_max": train_dataset.bbox_max.tolist() if train_dataset.bbox_max is not None else None,
    }

    for epoch in range(1, cfg.num_epochs + 1):
        policy.train()
        epoch_losses = []

        for batch in train_loader:
            batch = _batch_to_device(batch, device)

            loss, loss_dict = policy.compute_loss(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip_norm)
            optimizer.step()

            if ema_model is not None:
                ema_model.step(policy)

            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # ---- Validation ----
        val_loss = 0.0
        if epoch % cfg.val_every == 0:
            eval_policy = ema_model.averaged_model if ema_model is not None else policy
            val_loss = _evaluate(eval_policy, val_loader, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"[dp3][train] epoch={epoch:04d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # ---- Checkpointing ----
        # Checkpoints are written without optimizer state (inference-only) and
        # ``last.pt`` only every ``save_every_epochs`` (plus the final epoch).
        # A 255M-param model produces multi-GB checkpoints; writing one every
        # epoch saturates disk I/O and serializes parallel training jobs.
        ckpt_kwargs = dict(
            policy=policy,
            ema_model=ema_model,
            normalizer=normalizer,
            optimizer=optimizer,
            policy_cfg=policy_cfg,
            dataset_cfg=cfg.dataset,
            metadata=metadata,
            save_optimizer=False,
        )

        if epoch % cfg.save_every_epochs == 0 or epoch == cfg.num_epochs:
            _save_checkpoint(os.path.join(cfg.output_dir, "last.pt"), epoch=epoch, **ckpt_kwargs)

        if val_loss < best_val_loss and val_loss > 0:
            best_val_loss = val_loss
            best_epoch = epoch
            _save_checkpoint(os.path.join(cfg.output_dir, "best.pt"), epoch=epoch, **ckpt_kwargs)

    elapsed = time.time() - start_time
    summary = {
        "task": metadata["task"],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": history[-1]["train_loss"] if history else 0.0,
        "elapsed_s": elapsed,
    }

    # Save metrics
    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "config": dataclasses.asdict(cfg),
                "history": history,
                "summary": summary,
            },
            f,
            indent=2,
        )

    print(f"[dp3][train] Done. best_epoch={best_epoch} best_val_loss={best_val_loss:.6f} elapsed={elapsed:.1f}s")
    return summary
