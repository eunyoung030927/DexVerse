# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a parametric multi-level rack (shelf unit) as a single rigid USD.

Thin CLI around :func:`dexverse.assets.rack_builder.build_rack` (see that
module for the geometry / frame conventions and the manifest layout).

    python scripts/asset_tools/build_rack_usd.py                 # default 3-level rack
    python scripts/asset_tools/build_rack_usd.py --levels 4 --spacing 0.16 --name rack_4level

Requires ``pxr`` (ships with Isaac Sim's python). No simulator launch needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2] / "source" / "dexverse"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from dexverse.assets.rack_builder import RACK_ASSET_DIR, build_rack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path, default=RACK_ASSET_DIR)
    ap.add_argument("--name", type=str, default="rack_3level")
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--spacing", type=float, default=0.18, help="Vertical distance between shelf tops (m).")
    ap.add_argument("--depth", type=float, default=0.40, help="Rack extent along x (m).")
    ap.add_argument("--width", type=float, default=0.55, help="Rack extent along y (m).")
    ap.add_argument("--board_thickness", type=float, default=0.02)
    ap.add_argument("--post_size", type=float, default=0.03)
    ap.add_argument("--bottom_gap", type=float, default=0.10, help="Height of the lowest shelf top above the rack base (m).")
    ap.add_argument("--top_margin", type=float, default=0.03, help="Post height above the top shelf (m).")
    args = ap.parse_args()
    usd, manifest = build_rack(
        args.out_dir,
        name=args.name,
        levels=args.levels,
        spacing=args.spacing,
        depth=args.depth,
        width=args.width,
        board_thickness=args.board_thickness,
        post_size=args.post_size,
        bottom_gap=args.bottom_gap,
        top_margin=args.top_margin,
    )
    print(f"wrote {usd}\nwrote {manifest}")


if __name__ == "__main__":
    main()
