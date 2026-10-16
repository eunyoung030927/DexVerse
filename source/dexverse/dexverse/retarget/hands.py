# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-robot facts the cross-hand retargeting needs, keyed by ``robot_type`` (floating hands, single or bimanual).

A robot is one or two ``Hand``s (``right``, then ``left`` for bimanual robots). Each hand has a palm body, five
fingertip slots (thumb, index, middle, ring, little; ``None`` where the hand has no such finger) and six virtual
wrist joints. Across the robot, fingertip *slot* ``5 * hand + finger`` indexes every per-tip array (so a bimanual
source's left thumb is slot 5 and maps onto a bimanual target's left thumb), and ``palm_slot`` = hand index.
Body names are the ones the hand's own ``robot_agents/<hand>/floating.py`` declares.
"""

from __future__ import annotations

from dataclasses import dataclass

FINGERS = ("thumb", "index", "middle", "ring", "little")
SIDES = ("right", "left")

# wrist joint name patterns (translation x y z, rotation x y z), {s} = side where the names carry it
WRIST_XYZ = ("{s}x_translation_joint", "{s}y_translation_joint", "{s}z_translation_joint",
             "{s}x_rotation_joint", "{s}y_rotation_joint", "{s}z_rotation_joint")
WRIST_TRANS_ROT = ("{s}wrist_trans_x", "{s}wrist_trans_y", "{s}wrist_trans_z",
                   "{s}wrist_rot_x", "{s}wrist_rot_y", "{s}wrist_rot_z")
WRIST_RPY = ("{s}x_joint", "{s}y_joint", "{s}z_joint", "{s}roll_joint", "{s}pitch_joint", "{s}yaw_joint")


@dataclass(frozen=True)
class Hand:
    side: str
    palm: str
    tips: tuple[str | None, str | None, str | None, str | None, str | None]
    wrist: tuple[str, str, str, str, str, str]
    # underactuated hands: (joint, driver joint, multiplier, offset) -- joint = multiplier * driver + offset, the
    # URDF <mimic> that the USD keeps as a PhysX mimic constraint. Stage 2 solves only the drivers.
    mimic: tuple[tuple[str, str, float, float], ...] = ()


@dataclass(frozen=True)
class HandSpec:
    robot_type: str
    hands: tuple[Hand, ...]

    @property
    def n_hands(self) -> int:
        return len(self.hands)

    @property
    def palms(self) -> tuple[str, ...]:
        return tuple(h.palm for h in self.hands)

    @property
    def tips(self) -> tuple[str | None, ...]:
        """Fingertip bodies over all slots (5 per hand)."""
        return tuple(t for h in self.hands for t in h.tips)

    @property
    def wrist_joints(self) -> tuple[str, ...]:
        """Wrist joints of every hand, 6 per hand, hand-major."""
        return tuple(j for h in self.hands for j in h.wrist)

    @property
    def mimic(self) -> tuple[tuple[str, str, float, float], ...]:
        return tuple(m for h in self.hands for m in h.mimic)

    def tip_slots(self) -> list[int]:
        """Slots (5 * hand + finger) this robot has a fingertip for."""
        return [i for i, b in enumerate(self.tips) if b is not None]

    def slot_hand(self, slot: int) -> int:
        return slot // 5

    def slot_name(self, slot: int) -> str:
        return f"{self.hands[slot // 5].side}_{FINGERS[slot % 5]}" if self.n_hands > 1 else FINGERS[slot % 5]


def _w(pattern: tuple[str, ...], side_prefix: str) -> tuple[str, ...]:
    return tuple(p.format(s=side_prefix) for p in pattern)


def _single(robot_type, palm, tips, wrist, mimic=()) -> HandSpec:
    return HandSpec(robot_type, (Hand("right", palm, tuple(tips), tuple(wrist), tuple(mimic)),))


def _bimanual(robot_type, palms, tips_by_side, wrist_by_side, mimic_by_side=((), ())) -> HandSpec:
    return HandSpec(robot_type, tuple(Hand(s, palms[i], tuple(tips_by_side[i]), tuple(wrist_by_side[i]),
                                           tuple(mimic_by_side[i])) for i, s in enumerate(SIDES)))


_ALLEGRO = ("link_15_0_tip", "link_3_0_tip", "link_7_0_tip", "link_11_0_tip", None)   # thumb, index, middle, ring
_LEAP = ("thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3", None)
_SHADOW = ("thtip", "fftip", "mftip", "rftip", "lftip")
_SHARPA = ("thumb_fingertip", "index_fingertip", "middle_fingertip", "ring_fingertip", "pinky_fingertip")
_WUJI = tuple(f"finger{i}_tip_link" for i in range(1, 6))
_INSPIRE = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
# robot_agents/inspire/floating_inspire_hand_right/*.urdf <mimic> (6 drivers, 6 mimics; both hands alike)
_INSPIRE_MIMIC = (("thumb_intermediate_joint", "thumb_proximal_pitch_joint", 1.334, 0.0),
                  ("thumb_distal_joint", "thumb_proximal_pitch_joint", 0.667, 0.0),
                  *((f"{f}_intermediate_joint", f"{f}_proximal_joint", 1.06399, -0.04545)
                    for f in ("index", "middle", "ring", "pinky")))


def _pre(tips, prefix):
    return tuple(None if t is None else prefix + t for t in tips)


def _pre_mimic(mimic, prefix):
    return tuple((prefix + j, prefix + d, m, b) for j, d, m, b in mimic)


HANDS: dict[str, HandSpec] = {
    # ---- single right hands
    "floating_shadow_right": _single("floating_shadow_right", "palm", _SHADOW, _w(WRIST_XYZ, "")),
    "floating_allegro_right": _single("floating_allegro_right", "palm", _ALLEGRO, _w(WRIST_XYZ, "")),
    "floating_leap_right": _single("floating_leap_right", "palm_lower", _LEAP, _w(WRIST_XYZ, "")),
    "floating_wuji_right": _single("floating_wuji_right", "right_palm_link", _pre(_WUJI, "right_"), _w(WRIST_RPY, "right_")),
    "floating_sharpa_right": _single("floating_sharpa_right", "right_hand_C_MC", _pre(_SHARPA, "right_"), _w(WRIST_RPY, "right_")),
    "floating_inspire_right": _single("floating_inspire_right", "hand_base_link", _INSPIRE, _w(WRIST_XYZ, ""), _INSPIRE_MIMIC),
    # ---- bimanual (right, left)
    "floating_shadow_bimanual": _bimanual(
        "floating_shadow_bimanual", ("rh_palm", "lh_palm"), (_pre(_SHADOW, "rh_"), _pre(_SHADOW, "lh_")),
        (_w(WRIST_XYZ, "rh_"), _w(WRIST_XYZ, "lh_"))),
    "floating_allegro_bimanual": _bimanual(
        "floating_allegro_bimanual", ("right_palm", "left_palm"), (_pre(_ALLEGRO, "right_"), _pre(_ALLEGRO, "left_")),
        (_w(WRIST_TRANS_ROT, "right_"), _w(WRIST_TRANS_ROT, "left_"))),
    "floating_leap_bimanual": _bimanual(
        "floating_leap_bimanual", ("right_palm_lower", "left_palm_lower_left"), (_pre(_LEAP, "right_"), _pre(_LEAP, "left_")),
        (_w(WRIST_TRANS_ROT, "right_"), _w(WRIST_TRANS_ROT, "left_"))),
    "floating_wuji_bimanual": _bimanual(
        "floating_wuji_bimanual", ("right_palm_link", "left_palm_link"), (_pre(_WUJI, "right_"), _pre(_WUJI, "left_")),
        (_w(WRIST_RPY, "right_"), _w(WRIST_RPY, "left_"))),
    "floating_sharpa_bimanual": _bimanual(
        "floating_sharpa_bimanual", ("right_hand_C_MC", "left_hand_C_MC"), (_pre(_SHARPA, "right_"), _pre(_SHARPA, "left_")),
        (_w(WRIST_RPY, "right_"), _w(WRIST_RPY, "left_"))),
    "floating_inspire_bimanual": _bimanual(
        "floating_inspire_bimanual", ("right_hand_base_link", "left_hand_base_link"),
        (_pre(_INSPIRE, "right_"), _pre(_INSPIRE, "left_")), (_w(WRIST_TRANS_ROT, "right_"), _w(WRIST_TRANS_ROT, "left_")),
        (_pre_mimic(_INSPIRE_MIMIC, "right_"), _pre_mimic(_INSPIRE_MIMIC, "left_"))),
}


def get(robot_type: str) -> HandSpec:
    if robot_type not in HANDS:
        raise KeyError(f"no retargeting hand spec for '{robot_type}'; known: {sorted(HANDS)}")
    return HANDS[robot_type]
