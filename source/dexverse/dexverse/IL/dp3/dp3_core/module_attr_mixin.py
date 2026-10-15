# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from 3D-Diffusion-Policy (https://github.com/YanjieZe/3D-Diffusion-Policy)
# Original file: diffusion_policy_3d/model/common/module_attr_mixin.py
import torch.nn as nn


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
