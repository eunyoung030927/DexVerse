# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive of the six virtual wrist joints (x/y/z translation + three rotations) of every floating hand.

The virtual wrist joints stand in for the robot arm, so they should behave the same whichever hand is mounted.
Upstream shipped two drive sets: Shadow / Sharpa / Wuji used stiffness 2000, damping 400, effort limit 15, while
Allegro / Inspire / LEAP used 1000 / 200 / 50 (velocity limits differed too). All hands now share the Shadow
values, because the released demonstrations are Shadow demos recorded with them: replaying them open loop with
the Allegro set keeps only 62% of the successes (10/2026 replay study), while moving the Allegro / Inspire VR demos
to the Shadow set keeps 76-85%.

Set ``DEXVERSE_FLOATING_WRIST_DRIVE=per_hand`` before the robot configs are imported to restore each hand's
upstream values (e.g. to reproduce official benchmark numbers). A trajectory pickle must be replayed with the
drive it was recorded with.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

WRIST_DRIVE_ENV = "DEXVERSE_FLOATING_WRIST_DRIVE"


@dataclass(frozen=True)
class WristDrive:
    """PD drive of the virtual wrist joints (translation in N, m; rotation in Nm, rad)."""

    stiffness: float
    damping: float
    effort_limit: float
    velocity_limit_trans: float
    velocity_limit_rot: float
    friction: float = 0.01


SHADOW_WRIST_DRIVE = WristDrive(
    stiffness=2000.0, damping=400.0, effort_limit=15.0, velocity_limit_trans=10.0, velocity_limit_rot=5.0
)
"""The drive every floating hand uses (the values the released Shadow demonstrations were recorded with)."""

UPSTREAM_ALLEGRO_WRIST_DRIVE = WristDrive(
    stiffness=1000.0, damping=200.0, effort_limit=50.0, velocity_limit_trans=1.0, velocity_limit_rot=6.28
)
"""Upstream per-hand values of Allegro / Inspire / LEAP (used only with ``per_hand``)."""

UPSTREAM_SHARPA_WUJI_WRIST_DRIVE = WristDrive(
    stiffness=2000.0, damping=400.0, effort_limit=15.0, velocity_limit_trans=10.0, velocity_limit_rot=10.0
)
"""Upstream per-hand values of Sharpa / Wuji (used only with ``per_hand``)."""


def wrist_drive_mode() -> str:
    mode = os.environ.get(WRIST_DRIVE_ENV, "shared").strip().lower() or "shared"
    if mode not in ("shared", "per_hand"):
        raise ValueError(f"{WRIST_DRIVE_ENV}={mode!r}: expected 'shared' or 'per_hand'")
    return mode


def wrist_drive_terms(
    trans_expr: str, rot_expr: str, upstream: WristDrive = SHADOW_WRIST_DRIVE
) -> dict[str, dict[str, float]]:
    """``ImplicitActuatorCfg`` fields for the virtual wrist joints, keyed by joint-name expression.

    Args:
        trans_expr: Regex of the translation joints.
        rot_expr: Regex of the rotation joints.
        upstream: The hand's upstream drive, used instead of the shared one when ``per_hand`` is selected.

    Returns:
        ``{"effort_limit_sim": {...}, "stiffness": {...}, "damping": {...}, "velocity_limit_sim": {...},
        "friction": {...}}``; unpack into the actuator config or merge with the finger entries.
    """
    d = upstream if wrist_drive_mode() == "per_hand" else SHADOW_WRIST_DRIVE
    return {
        "effort_limit_sim": {trans_expr: d.effort_limit, rot_expr: d.effort_limit},
        "stiffness": {trans_expr: d.stiffness, rot_expr: d.stiffness},
        "damping": {trans_expr: d.damping, rot_expr: d.damping},
        "velocity_limit_sim": {trans_expr: d.velocity_limit_trans, rot_expr: d.velocity_limit_rot},
        "friction": {trans_expr: d.friction, rot_expr: d.friction},
    }
