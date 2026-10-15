# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a repository script with this checkout's dexverse package.

Example: python scripts/run_local.py scripts/list_envs.py --baseline --names_only
Reuses the active interpreter and dependencies without changing editable installs.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/run_local.py <repository-script.py> [arguments...]")
    script = (root / sys.argv[1]).resolve()
    if not script.is_relative_to(root) or not script.is_file():
        raise SystemExit(f"Expected a script inside {root}: {script}")
    source = root / "source" / "dexverse"
    sys.path[:0] = [str(source), str(script.parent)]
    os.environ["PYTHONPATH"] = str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")
    import dexverse

    if not Path(dexverse.__file__).resolve().is_relative_to(source):
        raise SystemExit(f"Wrong dexverse source: {dexverse.__file__}")
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
