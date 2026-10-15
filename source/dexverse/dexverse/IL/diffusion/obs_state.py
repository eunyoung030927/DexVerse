# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The policy's state vector: which observation groups it holds, in what order.

Training data and online evaluation must agree on this layout exactly. If they
disagree the policy is fed a permuted state vector at eval time and fails in a
way that looks like a training problem, so the layout lives here and both sides
import it:

* :func:`build_state` builds the vector from a live environment observation
  (used by the eval runner).
* ``scripts/diffusion/build_dataset.py`` rebuilds the same vector offline from a
  recorded HDF5, reading the group tuples below.

Every group listed here is declared with ``concatenate_terms = True``, so the
environment publishes one flat tensor per group and the vector is just those
tensors concatenated in the order below.
"""

from __future__ import annotations

from typing import Any

import torch

REQUIRED_GROUPS: tuple[str, ...] = ("policy", "proprio")
"""Groups every DexVerse task publishes. A missing one is a hard error."""

OPTIONAL_GROUPS: tuple[str, ...] = ("state", "privileged", "goal", "contact")
"""Task-dependent groups, included whenever the environment publishes them.

``state`` and ``privileged`` are where the object pose lives. Leaving them out
conditions the policy on the goal alone -- it can no longer tell where the
object is, which shows up as a hand that reaches for nothing.
"""


def build_state(obs: dict[str, Any], num_envs: int = 1) -> torch.Tensor:
    """Return the flat state vector for env 0 of a live observation dict."""
    if not isinstance(obs, dict):
        raise TypeError(f"Expected an observation dict, got {type(obs).__name__}.")

    chunks = []
    for name in REQUIRED_GROUPS:
        if name not in obs:
            raise KeyError(f"Observation is missing required group '{name}' (have: {sorted(obs)}).")
        chunks.append(_flatten_group(name, obs[name], num_envs))
    for name in OPTIONAL_GROUPS:
        if name in obs:
            chunks.append(_flatten_group(name, obs[name], num_envs))
    return torch.cat(chunks, dim=0)


def _flatten_group(name: str, value: Any, num_envs: int) -> torch.Tensor:
    if isinstance(value, dict):
        raise TypeError(
            f"Observation group '{name}' arrived as a per-term dict {sorted(value)}. The "
            "diffusion baseline requires concatenate_terms=True on every state group so "
            "that term order is fixed by the ObservationManager, not by dict iteration."
        )
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim >= 2 and tensor.shape[0] == num_envs:
        tensor = tensor[0]
    return tensor.reshape(-1)
