# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import collections

# Adapted from 3D-Diffusion-Policy (https://github.com/YanjieZe/3D-Diffusion-Policy)
# Original file: diffusion_policy_3d/common/pytorch_util.py
from typing import Callable, Dict, List

import torch
import torch.nn as nn


def dict_apply(x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def pad_remaining_dims(x, target):
    assert x.shape == target.shape[: len(x.shape)]
    return x.reshape(x.shape + (1,) * (len(target.shape) - len(x.shape)))


def dict_apply_split(
    x: Dict[str, torch.Tensor], split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results


def dict_apply_reduce(
    x: List[Dict[str, torch.Tensor]], reduce_func: Callable[[List[torch.Tensor]], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer
