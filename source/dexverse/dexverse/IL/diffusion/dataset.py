# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sliding-window dataset over training HDF5 files.

Input files are the ones written by ``scripts/diffusion/build_dataset.py``::

    data/<demo>/obs/state            (T, state_dim)
    data/<demo>/obs/images/<camera>  (T, 3, H, W) uint8, optional (image policy)
    data/<demo>/actions              (T, action_dim)

Each window is one timestep ``t0``: the ``obs_horizon`` states ending at ``t0``
(left-padded with the first state near the episode start) and the
``action_horizon`` actions starting at ``t0`` (right-padded with the last action
near the end, with a mask so padding does not contribute to the loss).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import h5py
import torch
from torch.utils.data import Dataset

from .config import DatasetConfig
from .normalization import Stats, minmax_stats


@dataclass
class Episode:
    name: str
    state: torch.Tensor
    action: torch.Tensor
    images: dict[str, torch.Tensor] = field(default_factory=dict)
    """Camera term name -> ``(T, 3, H, W)`` uint8 (kept uint8 in memory; the model scales to [0, 1])."""


class DiffusionDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        if cfg.split not in {"train", "val", "all"}:
            raise ValueError(f"split must be one of train/val/all, got '{cfg.split}'.")
        if cfg.obs_horizon <= 0 or cfg.action_horizon <= 0:
            raise ValueError("obs_horizon and action_horizon must both be > 0.")

        self.cfg = cfg
        episodes = _load_episodes(cfg.dataset_files)
        if cfg.state_index:
            index = torch.tensor(cfg.state_index, dtype=torch.long)
            for ep in episodes:
                ep.state = ep.state[:, index]
        self.episodes = [episodes[i] for i in _split_indices(len(episodes), cfg)]
        self.windows = [(idx, t0) for idx, ep in enumerate(self.episodes) for t0 in range(ep.action.shape[0])]
        if not self.windows:
            raise ValueError(f"Split '{cfg.split}' contains no windows.")

        self.stats: dict[str, Stats] = {
            "state": minmax_stats(torch.cat([ep.state for ep in self.episodes], dim=0)),
            "action": minmax_stats(torch.cat([ep.action for ep in self.episodes], dim=0)),
        }
        self.state_dim = int(self.episodes[0].state.shape[-1])
        self.action_dim = int(self.episodes[0].action.shape[-1])
        self.image_keys: tuple[str, ...] = tuple(self.episodes[0].images)
        self.image_size = int(next(iter(self.episodes[0].images.values())).shape[-1]) if self.image_keys else 0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_idx, t0 = self.windows[index]
        episode = self.episodes[episode_idx]
        length = episode.action.shape[0]

        obs_rows = [max(0, t) for t in range(t0 - self.cfg.obs_horizon + 1, t0 + 1)]
        action_rows = [min(length - 1, t0 + step) for step in range(self.cfg.action_horizon)]
        mask = [float(t0 + step < length) for step in range(self.cfg.action_horizon)]

        item = {
            "obs": episode.state[obs_rows],
            "action": episode.action[action_rows],
            "action_mask": torch.tensor(mask, dtype=torch.float32),
        }
        if episode.images:
            item["images"] = {key: frames[obs_rows] for key, frames in episode.images.items()}
        return item


def _load_episodes(dataset_files: tuple[str, ...]) -> list[Episode]:
    episodes: list[Episode] = []
    for path in dataset_files:
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise ValueError(f"'{path}' has no top-level 'data' group; not a training HDF5.")
            for name in sorted(handle["data"]):
                demo = handle["data"][name]
                state = torch.as_tensor(demo["obs/state"][...], dtype=torch.float32)
                action = torch.as_tensor(demo["actions"][...], dtype=torch.float32)
                length = min(state.shape[0], action.shape[0])
                images = {}
                if "obs/images" in demo:
                    # Sorted by name, so encoder order never depends on HDF5 iteration order.
                    images = {key: torch.from_numpy(demo["obs/images"][key][...]) for key in sorted(demo["obs/images"])}
                    length = min([length] + [frames.shape[0] for frames in images.values()])
                    images = {key: frames[:length] for key, frames in images.items()}
                if length > 0:
                    episodes.append(Episode(name, state[:length], action[:length], images))
    if not episodes:
        raise ValueError(f"No usable episodes in {list(dataset_files)}.")

    state_dims = {ep.state.shape[-1] for ep in episodes}
    action_dims = {ep.action.shape[-1] for ep in episodes}
    if len(state_dims) > 1 or len(action_dims) > 1:
        raise ValueError(f"Inconsistent dims across episodes: state {state_dims}, action {action_dims}.")
    cameras = {(tuple(ep.images), tuple({tuple(f.shape[1:]) for f in ep.images.values()})) for ep in episodes}
    if len(cameras) > 1:
        raise ValueError(f"Inconsistent cameras / image shapes across episodes: {cameras}.")
    return episodes


def _split_indices(count: int, cfg: DatasetConfig) -> list[int]:
    """Deterministic per-episode train/val split."""
    if cfg.split == "all" or count <= 1:
        return list(range(count))
    generator = torch.Generator().manual_seed(cfg.seed)
    shuffled = torch.randperm(count, generator=generator).tolist()
    train_count = max(1, min(count - 1, round(count * cfg.split_ratio)))
    if cfg.split == "train":
        return sorted(shuffled[:train_count])
    return sorted(shuffled[train_count:])
