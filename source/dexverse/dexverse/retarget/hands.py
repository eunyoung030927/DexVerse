# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-hand facts the cross-hand retargeting needs, keyed by ``robot_type`` (single floating hands).

Fingertips are always listed thumb, index, middle, ring, little; a hand without a finger has ``None`` in
that slot (a four-finger target simply has no target for the source's little finger). Body names are the
ones the hand's own ``robot_agents/<hand>/floating.py`` declares; the wrist is the six virtual joints, named
``x/y/z_translation_joint`` + ``x/y/z_rotation_joint`` (Shadow, Allegro, LEAP) or ``<side>_x/y/z_joint`` +
``<side>_roll/pitch/yaw_joint`` (Wuji, Sharpa).
"""

from __future__ import annotations

from dataclasses import dataclass

FINGERS = ("thumb", "index", "middle", "ring", "little")
WRIST_TRANS = ("x_translation_joint", "y_translation_joint", "z_translation_joint")
WRIST_ROT = ("x_rotation_joint", "y_rotation_joint", "z_rotation_joint")
WRIST_RIGHT = ("right_x_joint", "right_y_joint", "right_z_joint", "right_roll_joint", "right_pitch_joint", "right_yaw_joint")


@dataclass(frozen=True)
class HandSpec:
    robot_type: str
    palm: str
    tips: tuple[str | None, str | None, str | None, str | None, str | None]
    wrist: tuple[str, ...] = WRIST_TRANS + WRIST_ROT

    @property
    def wrist_joints(self) -> tuple[str, ...]:
        return self.wrist

    def tip_slots(self) -> list[int]:
        """Finger slots (0 = thumb .. 4 = little) this hand has a tip for."""
        return [i for i, b in enumerate(self.tips) if b is not None]


HANDS: dict[str, HandSpec] = {
    "floating_shadow_right": HandSpec("floating_shadow_right", "palm", ("thtip", "fftip", "mftip", "rftip", "lftip")),
    "floating_allegro_right": HandSpec(
        "floating_allegro_right", "palm", ("link_15_0_tip", "link_3_0_tip", "link_7_0_tip", "link_11_0_tip", None)
    ),
    "floating_leap_right": HandSpec(
        "floating_leap_right", "palm_lower", ("thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3", None)
    ),
    "floating_wuji_right": HandSpec(
        "floating_wuji_right", "right_palm_link", tuple(f"right_finger{i}_tip_link" for i in range(1, 6)), WRIST_RIGHT
    ),
    "floating_sharpa_right": HandSpec(
        "floating_sharpa_right", "right_hand_C_MC",
        tuple(f"right_{f}_fingertip" for f in ("thumb", "index", "middle", "ring", "pinky")), WRIST_RIGHT
    ),
}


def get(robot_type: str) -> HandSpec:
    if robot_type not in HANDS:
        raise KeyError(f"no retargeting hand spec for '{robot_type}'; known: {sorted(HANDS)}")
    return HANDS[robot_type]
