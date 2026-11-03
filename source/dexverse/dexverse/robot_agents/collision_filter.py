# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision filtering between the links of one hand of a bimanual articulation.

A bimanual robot is a single PhysX articulation, so ``enabled_self_collisions`` is the only switch for collisions
between its links: off, the right hand passes through the left; on, every link of a hand also collides with the
other links of the same hand. On the Inspire and Wuji bimanual USDs those intra-hand collision meshes overlap at
rest (2026-11-03: Wuji's palms spun at 14 rad/s, Inspire's fingers were pushed open), which is why the release
shipped them with self-collisions off. :func:`filter_intra_hand_collisions` keeps self-collisions on but tells
PhysX (``UsdPhysics.FilteredPairsAPI``) not to collide a link with the other links of the *same* hand, so only the
right-vs-left contacts remain.
"""

from __future__ import annotations

from isaaclab.sim.utils.queries import find_matching_prim_paths
from isaaclab.sim.utils.stage import get_current_stage
from pxr import Usd, UsdPhysics


def filter_intra_hand_collisions(prim_path: str, prefixes: tuple[str, ...] = ("right_", "left_")) -> dict[str, int]:
    """Filter collisions between links that share a hand prefix. Returns the number of links per prefix."""
    stage = get_current_stage()
    counts: dict[str, int] = {}
    for root in find_matching_prim_paths(prim_path):
        groups: dict[str, list[Usd.Prim]] = {p: [] for p in prefixes}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            for p in prefixes:
                if prim.GetName().startswith(p):
                    groups[p].append(prim)
        for p, links in groups.items():
            counts[p] = len(links)
            for link in links:
                api = UsdPhysics.FilteredPairsAPI.Apply(link)
                rel = api.GetFilteredPairsRel()
                for other in links:
                    if other.GetPath() != link.GetPath():
                        rel.AddTarget(other.GetPath())
    return counts
