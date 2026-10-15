# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from pathlib import Path


def prefer_workspace_source() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source_root = project_root / "source" / "dexverse"
    source_root_str = str(source_root)
    if source_root.is_dir() and source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)


prefer_workspace_source()
