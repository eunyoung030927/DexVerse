# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `dexverse` extension. They start
with `Dexverse-` in their name.

    python scripts/list_envs.py                        # full table
    python scripts/list_envs.py --baseline             # only the baseline-manifest tasks
    python scripts/list_envs.py --baseline --names_only  # bare ids, no simulator launch
"""

"""Parse arguments (before the simulator, so the cheap paths need no Isaac)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "dexverse"))

parser = argparse.ArgumentParser(description="List the registered Dexverse-* environments.")
parser.add_argument(
    "--baseline",
    action="store_true",
    help="Only the curated baseline tasks (source/dexverse/demonstrations/baseline_manifest.txt).",
)
parser.add_argument(
    "--names_only",
    action="store_true",
    help="Print bare task ids, one per line (script-friendly). With --baseline this needs no simulator launch.",
)
parser.add_argument(
    "--version",
    choices=["v0", "v1", "all"],
    default=None,
    help="Task version; --baseline defaults to v0, the full listing defaults to all.",
)
args_cli = parser.parse_args()


def _baseline_task_ids() -> list[str]:
    """Task ids from the versioned baseline catalogue."""
    from dexverse.benchmark import baseline_tasks

    return list(baseline_tasks(args_cli.version or "v0"))


if args_cli.baseline and args_cli.names_only:
    print("\n".join(_baseline_task_ids()))
    raise SystemExit(0)

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import dexverse.tasks  # noqa: F401
import gymnasium as gym
from prettytable import PrettyTable


def main():
    """Print all environments registered in `dexverse` extension."""
    baseline = set(_baseline_task_ids()) if args_cli.baseline else None
    task_specs = [
        s
        for s in gym.registry.values()
        if "Dexverse-" in s.id
        and (baseline is None or s.id in baseline)
        and (args_cli.version in (None, "all") or s.id.endswith("-" + args_cli.version))
    ]
    if args_cli.baseline:
        missing = sorted(baseline - {s.id for s in task_specs})
        if missing:
            print(f"[list_envs] WARNING: baseline tasks not registered: {missing}", flush=True)
        # keep manifest order
        order = {tid: i for i, tid in enumerate(_baseline_task_ids())}
        task_specs.sort(key=lambda s: order.get(s.id, 1e9))

    if args_cli.names_only:
        print("\n".join(s.id for s in task_specs), flush=True)
        return

    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Baseline Environments" if args_cli.baseline else "Available Environments in Isaac Lab"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    for index, task_spec in enumerate(task_specs):
        table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])

    print(table, flush=True)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
