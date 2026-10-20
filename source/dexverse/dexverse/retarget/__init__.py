# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-hand retargeting of recorded DexVerse demos (e.g. the released Shadow demos -> Allegro / LEAP / ...).

Stages (``scripts/retarget/``): 1 reference from the source episode's recorded states, 2 kinematic fingertip IK
for the target hand, 3 SPIDER physics refinement in the task, 4 trajectory pickle of the successful runs. See
``docs/retarget_cross_hand.md``.
"""
