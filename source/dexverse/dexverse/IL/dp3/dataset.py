# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DP3 dataset loader for DexVerse.

Loads HDF5 files produced by ``scripts/dp3/convert_demos_to_dp3.py`` and
provides windowed sequence sampling compatible with the DP3 policy's
``compute_loss`` and ``predict_action`` interfaces.

Each sample returns::

    {
        "obs": {
            "point_cloud": Tensor (horizon, num_points, 3),
            "agent_pos":   Tensor (horizon, proprio_dim),
        },
        "action": Tensor (horizon, action_dim),
    }
"""
from __future__ import annotations

import copy
from typing import Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DP3DatasetConfig
from .dp3_core.normalizer import LinearNormalizer


class DexVerseDP3Dataset(Dataset):
    """Dataset for DP3 training on DexVerse demonstrations.

    Loads all episodes into memory from an HDF5 file and provides windowed
    sequence sampling with edge-padding, matching the convention used by DP3.
    """

    def __init__(self, cfg: DP3DatasetConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.horizon = cfg.horizon
        self.pad_before = cfg.n_obs_steps - 1
        self.pad_after = cfg.n_action_steps - 1

        # Load all episodes from HDF5 into memory
        self.episodes, self.metadata = self._load_hdf5(cfg.hdf5_path)

        # Train/val split by episode
        self._all_episode_indices = list(range(len(self.episodes)))
        self._train_indices, self._val_indices = self._split_episodes(len(self.episodes), cfg.val_ratio, cfg.seed)
        self._active_indices = self._train_indices if split == "train" else self._val_indices

        # Build flat sample index: list of (episode_idx, start_t)
        self._sample_indices = self._build_sample_indices()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_points(self) -> int:
        return int(self.metadata.get("num_points", self.episodes[0]["point_cloud"].shape[1]))

    @property
    def proprio_dim(self) -> int:
        return int(self.episodes[0]["agent_pos"].shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.episodes[0]["action"].shape[-1])

    @property
    def bbox_min(self) -> np.ndarray | None:
        return self.metadata.get("bbox_min")

    @property
    def bbox_max(self) -> np.ndarray | None:
        return self.metadata.get("bbox_max")

    def get_normalizer(self, mode: str = "limits") -> LinearNormalizer:
        """Fit a LinearNormalizer on the active (train) split data."""
        all_pcd = []
        all_agent_pos = []
        all_action = []
        for idx in self._active_indices:
            ep = self.episodes[idx]
            all_pcd.append(ep["point_cloud"].reshape(-1, 3))
            all_agent_pos.append(ep["agent_pos"])
            all_action.append(ep["action"])

        data = {
            "point_cloud": np.concatenate(all_pcd, axis=0),
            "agent_pos": np.concatenate(all_agent_pos, axis=0),
            "action": np.concatenate(all_action, axis=0),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode)
        return normalizer

    def get_validation_dataset(self) -> "DexVerseDP3Dataset":
        """Return a copy of this dataset pointing to the validation split."""
        val_ds = copy.copy(self)
        val_ds.split = "val"
        val_ds._active_indices = self._val_indices
        val_ds._sample_indices = val_ds._build_sample_indices()
        return val_ds

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start_t = self._sample_indices[idx]
        ep = self.episodes[ep_idx]
        T_ep = ep["point_cloud"].shape[0]

        # Gather indices for the full horizon window, clamping at episode edges
        indices = []
        for t in range(start_t - self.pad_before, start_t - self.pad_before + self.horizon):
            indices.append(np.clip(t, 0, T_ep - 1))

        pcd = ep["point_cloud"][indices]  # (horizon, N, 3)
        agent_pos = ep["agent_pos"][indices]  # (horizon, D)
        action = ep["action"][indices]  # (horizon, Da)

        return {
            "obs": {
                "point_cloud": torch.from_numpy(pcd).float(),
                "agent_pos": torch.from_numpy(agent_pos).float(),
            },
            "action": torch.from_numpy(action).float(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_hdf5(hdf5_path: str) -> tuple[list[dict], dict]:
        """Load all episodes and metadata from HDF5."""
        episodes = []
        with h5py.File(hdf5_path, "r") as hf:
            metadata = {
                "task": hf.attrs.get("task", "unknown"),
                "num_points": int(hf.attrs.get("num_points", 0)),
                "bbox_min": hf.attrs.get("bbox_min"),
                "bbox_max": hf.attrs.get("bbox_max"),
            }
            for ep_name in sorted(hf["data"].keys()):
                ep = hf["data"][ep_name]
                episodes.append({
                    "point_cloud": ep["obs/point_cloud"][...],  # (T, N, 3)
                    "agent_pos": ep["obs/agent_pos"][...],  # (T, D)
                    "action": ep["action"][...],  # (T, Da)
                })
        if not episodes:
            raise ValueError(f"No episodes found in {hdf5_path}")
        return episodes, metadata

    @staticmethod
    def _split_episodes(n_episodes: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
        """Split episode indices into train/val sets."""
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_episodes)
        val_count = max(1, int(round(n_episodes * val_ratio)))
        # Ensure at least 1 training episode
        val_count = min(val_count, n_episodes - 1)
        val_indices = sorted(perm[:val_count].tolist())
        train_indices = sorted(perm[val_count:].tolist())
        return train_indices, val_indices

    def _build_sample_indices(self) -> list[tuple[int, int]]:
        """Build a flat list of (episode_idx, start_timestep) pairs.

        Each timestep in each active episode becomes one sample. The windowed
        __getitem__ handles edge-padding via clamping.
        """
        indices = []
        for ep_idx in self._active_indices:
            T_ep = self.episodes[ep_idx]["point_cloud"].shape[0]
            for t in range(T_ep):
                indices.append((ep_idx, t))
        return indices
