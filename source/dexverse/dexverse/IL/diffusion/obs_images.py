# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The policy's camera inputs: which images it sees and how they are resized.

Each camera is one observation term of the env's ``rgb`` group (``concatenate_terms
= False``, so the group arrives as a ``{term name: (num_envs, H, W, 3) uint8}``
dict). The term name is the camera's identity everywhere:

* ``scripts/diffusion/build_dataset.py`` stores every ``rgb`` term of the replay
  HDF5 as ``obs/images/<term>`` after :func:`resize_images`,
* the policy builds one image encoder per term name (``PolicyConfig.image_keys``),
* the eval runner looks each encoder's image up by the same name in the live
  observation (:func:`build_images`), so a camera can never be fed to another
  camera's encoder.

Training data and evaluation resize with this one function, so both see the same
pixels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

IMAGE_GROUP = "rgb"
"""Env observation group (and replay HDF5 group) holding the RGB camera terms."""


def resize_images(images: Any, size: int) -> torch.Tensor:
    """``(N, H, W, 3)`` uint8 -> ``(N, 3, size, size)`` uint8, area-averaged."""
    x = torch.as_tensor(np.asarray(images) if not torch.is_tensor(images) else images)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected (N, H, W, 3) RGB images, got shape {tuple(x.shape)}.")
    x = x.permute(0, 3, 1, 2).float()
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="area")
    return x.round().clamp(0, 255).to(torch.uint8)


def build_images(obs: dict[str, Any], keys: tuple[str, ...], size: int, num_envs: int = 1) -> dict[str, torch.Tensor]:
    """Per-camera ``(3, size, size)`` uint8 images for env 0, looked up by term name."""
    group = obs.get(IMAGE_GROUP)
    if not isinstance(group, dict):
        raise KeyError(
            f"Observation has no '{IMAGE_GROUP}' term dict (groups: {sorted(obs)}); "
            "evaluate an image policy with --observation_preset rgb (or the preset it was trained with)."
        )
    missing = [key for key in keys if key not in group]
    if missing:
        raise KeyError(f"Cameras {missing} the policy was trained on are not in the live '{IMAGE_GROUP}' group {sorted(group)}.")
    out = {}
    for key in keys:
        image = torch.as_tensor(group[key])
        if image.ndim == 4 and image.shape[0] == num_envs:
            image = image[:1]
        elif image.ndim == 3:
            image = image.unsqueeze(0)
        out[key] = resize_images(image.cpu(), size)[0]
    return out
