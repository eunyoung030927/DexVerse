# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Put the Inspire left / bimanual finger visuals back on their links.

The released ``floating_inspire_hand_left`` and ``floating_inspire_hand_bimanual`` USDs were converted with the
GLB node transform applied twice: the converter baked each visual GLB's node translation / rotation into the mesh
points and also kept it on the ``/meshes/<link>/<glb node>`` Xform above the mesh. The finger visuals therefore
render 10-13 cm away from the palm (the joints and collision shapes are correct, so physics is unaffected). The
right-hand USD was re-converted later and has no such Xform.

This script resets those Xforms to identity in the ``configuration/*_base.usd`` layer, in place. It is idempotent.
The USDs are downloaded assets (not tracked in git), so run it again after ``download_robot_agents.py`` re-extracts
the inspire bundle. Needs ``pxr`` (e.g. Isaac Sim's ``python.sh`` or ``isaaclab.sh -p``).

Examples
--------
    python scripts/asset_tools/fix_inspire_visual_xforms.py            # fix left + bimanual
    python scripts/asset_tools/fix_inspire_visual_xforms.py --check    # report only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

INSPIRE_DIR = Path(__file__).resolve().parents[2] / "source" / "dexverse" / "dexverse" / "robot_agents" / "inspire"
VARIANTS = ("floating_inspire_hand_left", "floating_inspire_hand_bimanual")


def doubled_visual_xforms(stage: Usd.Stage) -> list[Usd.Prim]:
    """``/meshes/<link>/<glb node>`` Xforms that hold a mesh and carry a non-identity transform."""
    meshes = stage.GetPrimAtPath("/meshes")
    found = []
    for link in meshes.GetChildren() if meshes else []:
        for node in link.GetChildren():
            if node.GetName() == "World" or not node.IsA(UsdGeom.Xformable):
                continue
            if not any(child.IsA(UsdGeom.Mesh) for child in node.GetChildren()):
                continue
            if UsdGeom.Xformable(node).GetLocalTransformation() != Gf.Matrix4d(1.0):
                found.append(node)
    return found


def fix_layer(base_usd: Path, check: bool) -> int:
    layer = Sdf.Layer.FindOrOpen(str(base_usd))
    stage = Usd.Stage.Open(layer)
    prims = doubled_visual_xforms(stage)
    for prim in prims:
        print(f"  {'found' if check else 'reset'} {prim.GetPath()}")
        if check:
            continue
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(type(op.Get())(0.0, 0.0, 0.0))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(type(op.Get())(1.0, 0.0, 0.0, 0.0))
            elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set(type(op.Get())(1.0, 1.0, 1.0))
            else:
                raise RuntimeError(f"{prim.GetPath()}: unexpected xformOp {op.GetOpName()}")
    if prims and not check:
        layer.Save()
        left = doubled_visual_xforms(Usd.Stage.Open(Sdf.Layer.FindOrOpen(str(base_usd))))
        if left:
            raise RuntimeError(f"{base_usd}: {len(left)} Xforms still not identity after the fix")
    return len(prims)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="Only report the offending Xforms.")
    args = parser.parse_args()
    for variant in VARIANTS:
        base_usd = INSPIRE_DIR / variant / "configuration" / f"{variant}_base.usd"
        if not base_usd.is_file():
            print(f"{variant}: {base_usd} missing (download the inspire bundle first), skipped")
            continue
        print(f"{variant}:")
        n = fix_layer(base_usd, args.check)
        print(f"  {n} Xform(s) {'need fixing' if args.check else 'reset'}" if n else "  already clean")


if __name__ == "__main__":
    main()
