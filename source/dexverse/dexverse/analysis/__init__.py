# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline (simulator-free) analysis utilities.

* :mod:`~dexverse.analysis.reset_audit` -- statistics, flags, plots and the
  markdown report for the reset-distribution audit produced by
  ``scripts/env_tools/sample_reset_distributions.py``. Pure numpy/matplotlib;
  importable without Isaac Sim so results can be re-plotted anywhere.
"""

from .reset_audit import (
    BASELINE_TASKS,
    TASK_ROLES,
    analyze_task_dir,
    compute_stats,
    flag_task,
    load_task_dir,
    plot_grid,
    plot_task,
    resolve_roles,
    write_report,
)

__all__ = [
    "BASELINE_TASKS",
    "TASK_ROLES",
    "analyze_task_dir",
    "compute_stats",
    "flag_task",
    "load_task_dir",
    "plot_grid",
    "plot_task",
    "resolve_roles",
    "write_report",
]
