# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-dimension min-max normalization to ``[-1, 1]``.

States and actions are normalized with statistics computed over the training
split and stored in the checkpoint, so evaluation reuses exactly the mapping the
model was trained under.
"""

from __future__ import annotations

import torch

Stats = dict[str, torch.Tensor]
_EPS = 1e-6


def minmax_stats(data: torch.Tensor) -> Stats:
    """Per-dimension min/max of a ``(N, dim)`` tensor."""
    if data.ndim != 2:
        raise ValueError(f"Expected a rank-2 tensor, got shape {tuple(data.shape)}.")
    return {"min": data.min(dim=0).values, "max": data.max(dim=0).values}


def normalize(value: torch.Tensor, stats: Stats) -> torch.Tensor:
    lo, hi = _bounds(value, stats)
    return ((value - lo) / (hi - lo).clamp_min(_EPS)) * 2.0 - 1.0


def denormalize(value: torch.Tensor, stats: Stats) -> torch.Tensor:
    lo, hi = _bounds(value, stats)
    return (value + 1.0) * 0.5 * (hi - lo).clamp_min(_EPS) + lo


def stats_to(stats: dict[str, Stats], device: torch.device | str) -> dict[str, Stats]:
    """Move a ``{"state": ..., "action": ...}`` stats mapping to ``device``."""
    return {key: {name: value.to(device) for name, value in inner.items()} for key, inner in stats.items()}


def _bounds(value: torch.Tensor, stats: Stats) -> tuple[torch.Tensor, torch.Tensor]:
    return stats["min"].to(value.device, value.dtype), stats["max"].to(value.device, value.dtype)
