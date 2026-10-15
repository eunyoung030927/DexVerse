# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural desk props: a staple strip and a counter-top dispenser.

:func:`build_staple_strip` writes one *dynamic* rigid body whose visual is a
single merged mesh of ``num_staples`` U-shaped staples (crown + two legs, wire
cross-section) standing crown-up in a row along local **+x**, and whose
collider is the convex hull of that mesh (effectively the strip's bounding box,
so grasps are stable and no thin wire features touch PhysX). Local frame:
origin at the centre of the row on the legs' bottom plane (z = 0 where the
strip stands), +z up (crowns on top).

Task configs call :func:`ensure_staple_strip_asset` at construction time,
which builds the file on first use (the assets tree is not tracked by git;
this module is).
"""

from __future__ import annotations

import json
from pathlib import Path

from .cut_props_builder import _AUTHORED_DIR, _add_box, _ensure_asset, _make_material, _new_stage

STAPLE_STRIP_ASSET_DIR = _AUTHORED_DIR / "staple_strip"
DISPENSER_ASSET_DIR = _AUTHORED_DIR / "dispenser"


def _make_metal_material(stage, name: str, mname: str, rgb: tuple[float, float, float]):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, f"/{name}/Looks/{mname}")
    sh = UsdShade.Shader.Define(stage, f"/{name}/Looks/{mname}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.9)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _box_geometry(points: list, counts: list, indices: list, center, size) -> None:
    """Append an axis-aligned box (8 verts, 6 quads, outward winding) to the buffers."""
    cx, cy, cz = center
    hx, hy, hz = (s / 2.0 for s in size)
    base = len(points)
    corners = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]
    points.extend(corners)
    faces = [
        (0, 3, 2, 1),  # bottom (-z)
        (4, 5, 6, 7),  # top (+z)
        (0, 1, 5, 4),  # -y
        (2, 3, 7, 6),  # +y
        (1, 2, 6, 5),  # +x
        (3, 0, 4, 7),  # -x
    ]
    for face in faces:
        counts.append(4)
        indices.extend(base + i for i in face)


def build_staple_strip(
    out_dir: Path,
    name: str = "staple_strip",
    num_staples: int = 100,
    crown_width: float = 0.012,
    leg_height: float = 0.008,
    wire: float = 0.0006,
    pitch: float = 0.0006,
    mass: float = 0.01,
    color: tuple[float, float, float] = (0.78, 0.79, 0.82),
) -> tuple[Path, Path]:
    """Write ``<name>.usda`` + ``<name>_manifest.json`` and return their paths.

    ``num_staples`` staples of ``pitch`` spacing form a row of length
    ``num_staples * pitch`` along +x; each staple is a crown bar of width
    ``crown_width`` (along y) at height ``leg_height`` and two legs down to
    z = 0, all with a ``wire`` square cross-section (default 26/6-like staples,
    slightly fat wire for visibility: 100 staples = 6 cm).
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    out_dir.mkdir(parents=True, exist_ok=True)
    usd_path = out_dir / f"{name}.usda"
    manifest_path = out_dir / f"{name}_manifest.json"
    stage = _new_stage(usd_path, name, mass=mass)
    mat = _make_metal_material(stage, name, "StapleMat", color)

    length = num_staples * pitch
    points: list = []
    counts: list = []
    indices: list = []
    t = wire * 0.92  # slightly thinner than the pitch so individual staples read as separate wires
    leg_y = (crown_width - wire) / 2.0
    for i in range(num_staples):
        x = -length / 2.0 + (i + 0.5) * pitch
        _box_geometry(points, counts, indices, (x, 0.0, leg_height - wire / 2.0), (t, crown_width, wire))
        for sy in (-1.0, 1.0):
            _box_geometry(
                points, counts, indices, (x, sy * leg_y, (leg_height - wire) / 2.0), (t, wire, leg_height - wire)
            )
    mesh = UsdGeom.Mesh.Define(stage, f"/{name}/staples")
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([Gf.Vec3f(-length / 2.0, -crown_width / 2.0, 0.0), Gf.Vec3f(length / 2.0, crown_width / 2.0, leg_height)])
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdGeom.Tokens.convexHull if hasattr(UsdGeom.Tokens, "convexHull") else "convexHull")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    stage.GetRootLayer().Save()

    manifest = {
        "name": name,
        "usd": usd_path.name,
        "frame": "origin at the row centre on the legs' bottom plane; row along +x, crowns up (+z)",
        "params": {
            "num_staples": num_staples,
            "crown_width": crown_width,
            "leg_height": leg_height,
            "wire": wire,
            "pitch": pitch,
            "mass": mass,
            "color": list(color),
        },
        "length": length,
        "half_extents": [length / 2.0, crown_width / 2.0, leg_height / 2.0],
        "center_local": [0.0, 0.0, leg_height / 2.0],
        "row_axis_local": [1.0, 0.0, 0.0],
        "crown_axis_local": [0.0, 0.0, 1.0],
        "mesh_prim": "staples",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return usd_path, manifest_path


def ensure_staple_strip_asset(name: str = "staple_strip", out_dir: Path | None = None, **params) -> tuple[Path, dict]:
    """Return ``(usd_path, manifest)`` for a staple strip, generating it if
    missing or if the stored parameters differ from ``params``."""
    return _ensure_asset(build_staple_strip, STAPLE_STRIP_ASSET_DIR, name, out_dir, params)


def _add_cylinder(stage, name: str, prim_name: str, center, radius: float, height: float, mat, collide: bool = True):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    cyl = UsdGeom.Cylinder.Define(stage, f"/{name}/{prim_name}")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2.0), Gf.Vec3f(radius, radius, height / 2.0)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*center))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
        cyl.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(mat)


def build_dispenser(
    out_dir: Path,
    name: str = "dispenser",
    tray_size: tuple[float, float, float] = (0.16, 0.16, 0.012),
    column_size: tuple[float, float, float] = (0.06, 0.06, 0.26),
    arm_section: tuple[float, float] = (0.03, 0.03),
    arm_overhang: float = 0.02,
    nozzle_radius: float = 0.008,
    nozzle_length: float = 0.03,
    spout_height: float = 0.16,
    target_disk_radius: float = 0.045,
    body_color: tuple[float, float, float] = (0.55, 0.57, 0.60),
    nozzle_color: tuple[float, float, float] = (0.20, 0.45, 0.85),
    target_color: tuple[float, float, float] = (0.15, 0.75, 0.25),
) -> tuple[Path, Path]:
    """Write ``<name>.usda`` + ``<name>_manifest.json`` for a counter-top dispenser.

    One kinematic body: a drip **tray** on the table, a **column** at its back
    (+x), an **arm** from the column top forward over the tray, and a
    **nozzle** hanging from the arm's front end whose bottom sits
    ``spout_height`` above the tray top, exactly over the tray centre -- the
    **spot** (``manifest["spot_local"]``) where a cup is to be placed. A green
    visual-only **target disk** on the tray marks the spot (a camera-visible
    landmark). Local frame: origin at the tray centre on the table (z = 0),
    +z up, the open side (where a cup comes in) faces **-x**.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    usd_path = out_dir / f"{name}.usda"
    manifest_path = out_dir / f"{name}_manifest.json"
    stage = _new_stage(usd_path, name, mass=4.0)
    mat_body = _make_material(stage, name, "BodyMat", body_color)
    mat_nozzle = _make_material(stage, name, "NozzleMat", nozzle_color)
    mat_target = _make_material(stage, name, "TargetMat", target_color)
    tx, ty, tz = tray_size
    cx, cy, cz = column_size
    ax, az = arm_section
    tray_top = tz
    column_x = tx / 2.0 - cx / 2.0  # column flush with the tray's back edge
    _add_box(stage, name, "tray", (0.0, 0.0, tz / 2.0), tray_size, mat_body)
    _add_box(stage, name, "column", (column_x, 0.0, tray_top + cz / 2.0), column_size, mat_body)
    # arm: from the column's front face to just past the nozzle (at x = 0)
    arm_x0 = column_x - cx / 2.0
    arm_x1 = -arm_overhang
    arm_len = arm_x0 - arm_x1
    arm_z = tray_top + spout_height + nozzle_length + az / 2.0
    _add_box(stage, name, "arm", ((arm_x0 + arm_x1) / 2.0, 0.0, arm_z), (arm_len, ax, az), mat_body)
    nozzle_z = tray_top + spout_height + nozzle_length / 2.0
    _add_cylinder(stage, name, "nozzle", (0.0, 0.0, nozzle_z), nozzle_radius, nozzle_length, mat_nozzle)
    # visual-only target disk on the tray (no collision: a cup stands on the tray)
    _add_cylinder(stage, name, "target_disk", (0.0, 0.0, tray_top + 0.0005), target_disk_radius, 0.001, mat_target, collide=False)
    stage.GetRootLayer().Save()
    manifest = {
        "name": name,
        "usd": usd_path.name,
        "frame": "origin at the tray centre on the table; +z up; open side faces -x (column at +x)",
        "params": {
            "tray_size": list(tray_size),
            "column_size": list(column_size),
            "arm_section": list(arm_section),
            "arm_overhang": arm_overhang,
            "nozzle_radius": nozzle_radius,
            "nozzle_length": nozzle_length,
            "spout_height": spout_height,
            "target_disk_radius": target_disk_radius,
            "body_color": list(body_color),
            "nozzle_color": list(nozzle_color),
            "target_color": list(target_color),
        },
        "tray_top": tray_top,
        "spot_local": [0.0, 0.0, tray_top],
        "spout_bottom_local": [0.0, 0.0, tray_top + spout_height],
        "arm_bottom_z": tray_top + spout_height + nozzle_length,
        "footprint_half": [tx / 2.0, ty / 2.0],
        "column_front_x": arm_x0,
        "height": tray_top + cz,
        "target_disk_radius": target_disk_radius,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return usd_path, manifest_path


def ensure_dispenser_asset(name: str = "dispenser", out_dir: Path | None = None, **params) -> tuple[Path, dict]:
    """Return ``(usd_path, manifest)`` for a dispenser prop, generating it if
    missing or if the stored parameters differ from ``params``."""
    return _ensure_asset(build_dispenser, DISPENSER_ASSET_DIR, name, out_dir, params)
