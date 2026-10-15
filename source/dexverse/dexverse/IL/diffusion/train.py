# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline DDPM training loop for the diffusion baseline.

Pure PyTorch -- no Isaac Sim, so ``CUDA_VISIBLE_DEVICES`` is the way to pin a
GPU. Writes ``last.pt``, periodic ``epoch_*.pt``, the best-validation ``best.pt``
(holding the EMA weights) and ``metrics.json`` into ``output_dir``.

Under ``torchrun`` (``WORLD_SIZE`` > 1) training is DistributedDataParallel: each
rank takes ``batch_size / world_size`` samples so the global batch is unchanged,
and rank 0 validates, logs and writes checkpoints.
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import DatasetConfig, PolicyConfig, TrainConfig
from .dataset import DiffusionDataset
from .model import DiffusionPolicy
from .normalization import normalize, stats_to


def train(cfg: TrainConfig) -> dict[str, Any] | None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
        if cfg.batch_size % world_size:
            raise ValueError(f"--batch_size {cfg.batch_size} is not divisible by world size {world_size}")
    else:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(cfg.seed)

    dataset_kwargs = {
        "dataset_files": tuple(cfg.dataset_files),
        "obs_horizon": cfg.obs_horizon,
        "action_horizon": cfg.action_horizon,
        "seed": cfg.seed,
        "split_ratio": cfg.split_ratio,
    }
    train_set = DiffusionDataset(DatasetConfig(split="all" if cfg.no_val else "train", **dataset_kwargs))
    val_set = None if cfg.no_val else DiffusionDataset(DatasetConfig(split="val", **dataset_kwargs))

    policy_cfg = PolicyConfig(
        state_dim=train_set.state_dim,
        action_dim=train_set.action_dim,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        obs_embed_dim=cfg.obs_embed_dim,
        unet_block_out_channels=tuple(cfg.unet_block_out_channels),
        layers_per_block=cfg.layers_per_block,
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
    )
    model = DiffusionPolicy(policy_cfg).to(device)
    ema_model = copy.deepcopy(model).requires_grad_(False).eval()
    scheduler = model.make_noise_scheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_model = DistributedDataParallel(model, device_ids=[device.index]) if distributed else model

    per_rank_batch = cfg.batch_size // world_size
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=cfg.seed) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=per_rank_batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        drop_last=len(train_set) >= cfg.batch_size,
    )
    val_loader = None if val_set is None else DataLoader(val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    stats = stats_to(train_set.stats, device)
    if is_main:
        print(
            f"[diffusion] task={cfg.task} state_dim={policy_cfg.state_dim} action_dim={policy_cfg.action_dim} "
            f"episodes={len(train_set.episodes)}/{len(val_set.episodes) if val_set else 0} "
            f"windows={len(train_set)}/{len(val_set) if val_set else 0} "
            f"world_size={world_size} batch={per_rank_batch}x{world_size}",
            flush=True,
        )

    history: list[dict[str, float]] = []
    best_val_loss, best_epoch, epochs_since_best = float("inf"), -1, 0
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        losses = []
        for batch in train_loader:
            loss = _denoising_loss(train_model, batch, scheduler, stats, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            _ema_update(ema_model, model, cfg.ema_decay)
            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else 0.0
        if distributed:
            reduced = torch.tensor(train_loss, device=device)
            dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
            train_loss = reduced.item()
        stop = torch.zeros((), device=device)
        if not is_main:
            if distributed:
                dist.broadcast(stop, src=0)
            if stop.item():
                break
            continue
        val_loss = float("nan") if val_loader is None else _validate(ema_model, val_loader, scheduler, stats, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[diffusion] epoch {epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)

        checkpoint = {
            "task": cfg.task,
            "epoch": epoch,
            "policy_cfg": asdict(policy_cfg),
            "train_cfg": asdict(cfg),
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "stats": stats_to(train_set.stats, "cpu"),
        }
        if cfg.slim_checkpoints:
            # runner.load_policy reads ``ema_model`` when present, else ``model``.
            checkpoint["model"] = checkpoint.pop("ema_model")
        if cfg.save_last:
            _atomic_save(checkpoint, output_dir / "last.pt")
        if cfg.save_every_epochs > 0 and epoch % cfg.save_every_epochs == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:04d}.pt")

        if val_loader is None:
            pass
        elif val_loss < best_val_loss:
            best_val_loss, best_epoch, epochs_since_best = val_loss, epoch, 0
            if cfg.save_best:
                # Evaluation loads ``model``, so best.pt carries the EMA weights there.
                best = checkpoint if cfg.slim_checkpoints else {**checkpoint, "model": ema_model.state_dict()}
                _atomic_save(best, output_dir / "best.pt")
        else:
            epochs_since_best += 1
            if 0 < cfg.early_stopping_patience <= epochs_since_best:
                print(f"[diffusion] early stop at epoch {epoch}; best was {best_epoch}", flush=True)
                stop.fill_(1)
        if distributed:
            dist.broadcast(stop, src=0)
        if stop.item():
            break

    if not is_main:
        dist.destroy_process_group()
        return None

    metrics = {
        "task": cfg.task,
        "state_dim": policy_cfg.state_dim,
        "action_dim": policy_cfg.action_dim,
        "train_windows": len(train_set),
        "val_windows": len(val_set) if val_set else 0,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "elapsed_s": time.time() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps({"config": asdict(cfg), "policy_cfg": asdict(policy_cfg), "history": history, "metrics": metrics}, indent=2)
    )
    if distributed:
        dist.destroy_process_group()
    return metrics


def _atomic_save(obj, path: Path) -> None:
    """Write then rename, so a reader (e.g. an evaluation started mid-training) never sees a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _denoising_loss(model, batch, scheduler, stats, device) -> torch.Tensor:
    obs = normalize(batch["obs"].to(device), stats["state"])
    action = normalize(batch["action"].to(device), stats["action"])
    mask = batch["action_mask"].to(device).unsqueeze(-1)

    noise = torch.randn_like(action)
    timesteps = torch.randint(0, getattr(model, "module", model).cfg.num_train_timesteps, (action.shape[0],), device=device)
    predicted = model(scheduler.add_noise(action, noise, timesteps), timesteps, obs)

    error = F.mse_loss(predicted, noise, reduction="none") * mask
    return error.sum() / mask.expand_as(error).sum().clamp_min(1.0)


@torch.no_grad()
def _validate(model, loader, scheduler, stats, device) -> float:
    model.eval()
    losses = [_denoising_loss(model, batch, scheduler, stats, device).item() for batch in loader]
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def _ema_update(ema_model, model, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
