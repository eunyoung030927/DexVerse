# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural small side-table generator.

The table is one kinematic rigid body made of box prims -- a rectangular top
slab on four corner legs -- used as an object stand (e.g. the hammer-strike
hammer stand) so an episode starts with a furniture affordance rather than the
object lying on the main tabletop. Next to the ``.usda`` a manifest JSON
records the top-face height (``z_top``) and usable top extents in the table's
local frame, so task configs can seat objects on the top analytically.

Table local frame: origin at the centre of the footprint on the *bottom* face
(z = 0 is where the table stands on the main tabletop); +x = depth, +y = width,
+z = up. ``z_top`` equals the overall ``height``.

Task configs call :func:`ensure_small_table_asset` at construction time, which
builds the files on first use (the assets tree is not tracked by git).
"""

from __future__ import annotations

import json
from pathlib import Path

SMALL_TABLE_ASSET_DIR = Path(__file__).resolve().parent / "core_assets" / "dexverse_authored" / "small_table"


def _param_values_match(stored, requested) -> bool:
    """Compare cached scalar or sequence-valued builder parameters."""
    if isinstance(requested, (list, tuple)):
        return (
            isinstance(stored, (list, tuple))
            and len(stored) == len(requested)
            and all(_param_values_match(old, new) for old, new in zip(stored, requested, strict=True))
        )
    try:
        return abs(float(stored) - float(requested)) < 1e-9
    except (TypeError, ValueError):
        return stored == requested


def build_small_table(
    out_dir: Path,
    name: str = "small_table",
    depth: float = 0.40,
    width: float = 0.40,
    height: float = 0.06,
    top_thickness: float = 0.012,
    leg_size: float = 0.022,
    color: tuple[float, float, float] = (0.55, 0.42, 0.28),
) -> tuple[Path, Path]:
    """Write ``<name>.usda`` + ``<name>_manifest.json`` and return their paths."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    out_dir.mkdir(parents=True, exist_ok=True)
    usd_path = out_dir / f"{name}.usda"
    manifest_path = out_dir / f"{name}_manifest.json"

    leg_height = height - top_thickness

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(3.0)
    root.GetPrim().CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)

    # material
    UsdGeom.Scope.Define(stage, f"/{name}/Looks")
    mat = UsdShade.Material.Define(stage, f"/{name}/Looks/TableMat")
    shader = UsdShade.Shader.Define(stage, f"/{name}/Looks/TableMat/PreviewSurface")
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
    p = leg_size / 2.0
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        add_box(f"leg_{i}", (sx * (hx - p), sy * (hy - p), leg_height / 2.0), (leg_size, leg_size, leg_height))
    add_box("top", (0.0, 0.0, height - top_thickness / 2.0), (depth, width, top_thickness))
    stage.GetRootLayer().Save()

    manifest = {
        "name": name,
        "usd": usd_path.name,
        "frame": "origin at footprint centre on the bottom face; +x depth, +y width, +z up",
        "params": {
            "depth": depth,
            "width": width,
            "height": height,
            "top_thickness": top_thickness,
            "leg_size": leg_size,
            "color": color,
        },
        "footprint": {"x_half": hx, "y_half": hy, "height": height},
        "z_top": height,
        "top": {"x_range": [-hx, hx], "y_range": [-hy, hy]},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return usd_path, manifest_path


def ensure_small_table_asset(name: str = "small_table", out_dir: Path | None = None, **params) -> tuple[Path, dict]:
    """Return ``(usd_path, manifest)`` for a small table, generating it if
    missing or if the stored parameters differ from ``params``."""
    out_dir = Path(out_dir) if out_dir is not None else SMALL_TABLE_ASSET_DIR
    usd_path = out_dir / f"{name}.usda"
    manifest_path = out_dir / f"{name}_manifest.json"
    if usd_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        stored = manifest.get("params", {})
        if all(k in stored and _param_values_match(stored[k], v) for k, v in params.items()):
            return usd_path, manifest
    usd_path, manifest_path = build_small_table(out_dir, name=name, **params)
    return usd_path, json.loads(manifest_path.read_text())
