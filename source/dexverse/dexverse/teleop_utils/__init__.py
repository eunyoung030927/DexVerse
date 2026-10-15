# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Operator-facing teleoperation aids.

Two features, both aimed at giving the teleoperator information they otherwise
only get after the session is over:

* :mod:`~dexverse.teleop_utils.range_vis` outlines the environment's
  randomization ranges (object spawn envelopes, goal sampling boxes) as
  wireframe boxes in the scene, so the operator can see the space the task
  samples from rather than inferring it one episode at a time.
* :mod:`~dexverse.teleop_utils.stats_panel` re-scores demonstration diversity
  after every recorded trajectory and shows it as an in-headset panel, backed by
  the simulator-free metrics in :mod:`~dexverse.teleop_utils.diversity_core`.

``diversity_core`` is pure numpy and imports nothing from Isaac Sim, and
``stats_panel`` keeps its Kit/USD imports inside functions -- both are usable
offline. ``range_vis`` subclasses an Isaac Lab manager base class, so importing
it requires the Isaac Sim runtime; it is deliberately *not* re-exported here so
that this package stays importable without a simulator.
"""

__all__ = [
    "analyze_demos",
    "analyze_task",
    "demos_from_episodes",
    "load_task_demos",
]


def __getattr__(name):
    """Load optional analysis helpers only when explicitly requested.

    Task runtime helpers such as door latch progress must be importable without
    pulling in the collection/analysis utilities through this package initializer.
    """
    if name in __all__:
        from . import diversity_core

        value = getattr(diversity_core, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
