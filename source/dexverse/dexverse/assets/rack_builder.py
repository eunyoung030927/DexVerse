# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural multi-level rack (shelf unit) generator.

The rack is one kinematic rigid body made of box prims -- four slim corner
posts plus ``levels`` shelf boards -- open at the front and on both sides so two
hands can take a tray by its side edges. Next to the ``.usda`` a manifest JSON
records every level's frame (shelf-top height, usable x/y extents, clearance to
the level above) in the rack's local frame, so task configs can place objects
on a chosen level analytically.

Rack local frame: origin at the centre of the footprint on the *bottom* face
(z = 0 is where the rack stands on the table); +x = depth (the open front is the
-x face, i.e. facing the robot when the rack stands at +x on the table); +y =
width; +z = up.

Task configs call :func:`ensure_rack_asset` at construction time, which builds
the files on first use (the assets tree is not tracked by git). The CLI wrapper
is ``scripts/asset_tools/build_rack_usd.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

RACK_ASSET_DIR = Path(__file__).resolve().parent / "core_assets" / "dexverse_authored" / "rack"


def build_rack(
    out_dir: Path,
    name: str = "rack_3level",
    levels: int = 3,
    spacing: float = 0.18,
    depth: float = 0.40,
    width: float = 0.55,
    board_thickness: float = 0.02,
    post_size: float = 0.03,
    bottom_gap: float = 0.10,
    top_margin: float = 0.03,
    color: tuple[float, float, float] = (0.55, 0.42, 0.28),
) -> tuple[Path, Path]:
    """Write ``<name>.usda`` + ``<name>_manifest.json`` and return their paths."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    out_dir.mkdir(parents=True, exist_ok=True)
    usd_path = out_dir / f"{name}.usda"
    manifest_path = out_dir / f"{name}_manifest.json"

    # Level k shelf board occupies [z_k - t, z_k] with z_k its top face.
    level_tops = [bottom_gap + k * spacing for k in range(levels)]
    # Posts run from the table to a little above the top shelf.
    post_height = level_tops[-1] + top_margin

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(5.0)
    root.GetPrim().CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)

    # material
    UsdGeom.Scope.Define(stage, f"/{name}/Looks")
    mat = UsdShade.Material.Define(stage, f"/{name}/Looks/RackMat")
    shader = UsdShade.Shader.Define(stage, f"/{name}/Looks/RackMat/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    def add_box(prim_name: str, center: tuple[float, float, float], size: tuple[float, float, float]) -> None:
        cube = UsdGeom.Cube.Define(stage, f"/{name}/{prim_name}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*center))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)

    hx, hy = depth / 2.0, width / 2.0
    p = post_size / 2.0
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        add_box(f"post_{i}", (sx * (hx - p), sy * (hy - p), post_height / 2.0), (post_size, post_size, post_height))
    for k, z_top in enumerate(level_tops):
        add_box(
            f"shelf_{k}",
            (0.0, 0.0, z_top - board_thickness / 2.0),
            (depth - 2 * post_size, width - 2 * post_size, board_thickness),
        )
    stage.GetRootLayer().Save()

    manifest = {
        "name": name,
        "usd": usd_path.name,
        "frame": "origin at footprint centre on the bottom face; +x depth (open front is -x), +y width, +z up",
        "params": {
            "levels": levels,
            "spacing": spacing,
            "depth": depth,
            "width": width,
            "board_thickness": board_thickness,
            "post_size": post_size,
            "bottom_gap": bottom_gap,
            "top_margin": top_margin,
        },
        "footprint": {"x_half": hx, "y_half": hy, "height": post_height},
        "levels": [
            {
                "index": k,
                "z_top": z_top,
                "clearance": (level_tops[k + 1] - board_thickness - z_top) if k + 1 < levels else None,
                "x_range": [-(hx - post_size), (hx - post_size)],
                "y_range": [-(hy - post_size), (hy - post_size)],
            }
            for k, z_top in enumerate(level_tops)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return usd_path, manifest_path




def ensure_rack_asset(name: str = "rack_3level", out_dir: Path | None = None, **params) -> tuple[Path, dict]:
    """Return ``(usd_path, manifest)`` for a rack, generating it if missing or
    if the stored parameters differ from ``params``."""
    out_dir = Path(out_dir) if out_dir is not None else RACK_ASSET_DIR
    usd_path = out_dir / f"{name}.usda"
    manifest_path = out_dir / f"{name}_manifest.json"
    if usd_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        stored = manifest.get("params", {})
        if all(abs(float(stored.get(k, float("nan"))) - float(v)) < 1e-9 for k, v in params.items()):
            return usd_path, manifest
    usd_path, manifest_path = build_rack(out_dir, name=name, **params)
    return usd_path, json.loads(manifest_path.read_text())
