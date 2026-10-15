# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Baseline-v1 control overrides; robot assets and actuator definitions stay shared.

These preserve the v1 collection/control setup, not v0 policy compatibility.
Always copy nested configuration before overriding it so v0 is unaffected.
"""

from copy import deepcopy
import math

SHADOW_ROBOT_TYPES = frozenset({
    "floating_shadow_right", "floating_shadow_left", "floating_shadow_bimanual",
})
SHADOW_WRIST_LIMITS = (-2.0 * math.pi, 2.0 * math.pi)
VECTOR_LOW_PASS_ALPHA = 0.2
VECTOR_SCALING_FACTOR = 1.125


def action_layout(layout: dict) -> dict:
    """Copy a shared Shadow layout and retain the v1 wrist command limits."""
    result = deepcopy(layout)
    for hand in result["hands"].values():
        hand["wrist_euler_limits"] = (SHADOW_WRIST_LIMITS,) * 3
    return result


def retargeting_config(config: dict, scheme: str) -> dict:
    """Copy the shared YAML settings; only vector collection settings differ."""
    result = deepcopy(config)
    if scheme == "vector":
        result["retargeting"].update(
            low_pass_alpha=VECTOR_LOW_PASS_ALPHA, scaling_factor=VECTOR_SCALING_FACTOR,
        )
    return result
