# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural single-slot shelf ("cubby with one gap") generator.

A minimalist shelf for insertion tasks: base + back + two side walls, plus the
slot-forming blocks. Two orientations:

- ``slot_orientation="vertical"`` (default): two filler regions flank a single
  vertical slot. They can be solid blocks, or short individual folder props
  when ``vertical_filler_style="folder_props"``. In both cases the centre slot
  remains the intended opening and the object stands in it like a file.
- ``slot_orientation="horizontal"``: one solid block fills the cavity from the
  base up to ``slot_bottom_height``; its top face is the slot floor (the slot's
  only, *bottom* limit -- there is deliberately no upper board), and the object
  is laid flat on it through the open front / top.

Open at the front (-x) and, by default, at the top. One kinematic rigid USD;
``<name>_manifest.json`` records the slot frame in the shelf's local frame so a
task can test "object inside the slot" analytically (``width_axis_local`` is
the shelf-local axis the inserted object's thickness axis must align with:
``+y`` for the vertical slot, ``+z`` -- i.e. lying flat -- for the horizontal).

Shelf local frame: origin at the centre of the footprint on the *bottom* face
(z = 0 where it stands), +x depth (open front is -x), +y width, +z up.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SHELF_ASSET_DIR = Path(__file__).resolve().parent / "core_assets" / "dexverse_authored" / "shelf"


def build_slot_shelf(
    out_dir: Path,
    name: str = "slot_shelf",
    inner_depth: float = 0.30,
    slot_width: float = 0.09,
    filler_width: float = 0.20,
    height: float = 0.32,
    panel_thickness: float = 0.02,
    top_closed: bool = False,
    slot_orientation: str = "vertical",
    vertical_filler_style: str = "solid",
    slot_bottom_height: float = 0.10,
    color: tuple[float, float, float] = (0.55, 0.42, 0.28),
    filler_color: tuple[float, float, float] = (0.36, 0.30, 0.26),
) -> tuple[Path, Path]:
    """Write ``<name>.usda`` + ``<name>_manifest.json`` and return their paths."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    if slot_orientation not in ("vertical", "horizontal"):
        raise ValueError(f"slot_orientation must be 'vertical' or 'horizontal', got {slot_orientation!r}")
    if vertical_filler_style not in ("solid", "folder_props"):
        raise ValueError(
            "vertical_filler_style must be 'solid' or 'folder_props', "
            f"got {vertical_filler_style!r}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    usd_path = out_dir / f"{name}.usda"
    manifest_path = out_dir / f"{name}_manifest.json"

    t = panel_thickness
    depth = inner_depth + t  # + back panel
    width = 2 * t + 2 * filler_width + slot_width
    hx, hy = depth / 2.0, width / 2.0
    inner_h = height - t  # above the base board

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(8.0)
    root.GetPrim().CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)

    UsdGeom.Scope.Define(stage, f"/{name}/Looks")

    def make_material(mname, rgb):
        mat = UsdShade.Material.Define(stage, f"/{name}/Looks/{mname}")
        sh = UsdShade.Shader.Define(stage, f"/{name}/Looks/{mname}/PreviewSurface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    mat_frame = make_material("ShelfMat", color)
    mat_fill = make_material("FillerMat", filler_color)
    if vertical_filler_style == "folder_props":
        folder_materials = (
            make_material("FolderBlue", (0.24, 0.39, 0.58)),
            make_material("FolderRust", (0.62, 0.30, 0.22)),
            make_material("FolderGreen", (0.28, 0.48, 0.36)),
        )
        mat_label = make_material("FolderLabel", (0.84, 0.81, 0.68))
    else:
        folder_materials = ()
        mat_label = None

    def add_box(prim_name, center, size, mat, parent_path=None, collision=True):
        parent_path = parent_path or f"/{name}"
        cube = UsdGeom.Cube.Define(stage, f"{parent_path}/{prim_name}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*center))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        if collision:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            cube.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)

    def add_folder_prop(prim_name, center, size, lean_x_deg, mat):
        """Add one collidable file folder plus a small visual front label."""
        if mat_label is None:
            raise RuntimeError("Folder props require the folder-prop materials")
        folder_path = f"/{name}/{prim_name}"
        folder = UsdGeom.Xform.Define(stage, folder_path)
        folder_xf = UsdGeom.Xformable(folder.GetPrim())
        folder_xf.AddTranslateOp().Set(Gf.Vec3d(*center))
        folder_xf.AddRotateXYZOp().Set(Gf.Vec3f(lean_x_deg, 0.0, 0.0))
        depth, thickness, folder_height = size
        add_box("body", (0.0, 0.0, 0.0), size, mat, parent_path=folder_path)
        # A recessed paper label on the visible front edge makes the cuboid read
        # as a stored folder without adding another collision surface.
        add_box(
            "label",
            (-depth / 2.0 - 0.0011, 0.0, -folder_height / 2.0 + 0.065),
            (0.002, thickness * 0.68, 0.045),
            mat_label,
            parent_path=folder_path,
            collision=False,
        )

    # frame
    add_box("base", (0.0, 0.0, t / 2.0), (depth, width, t), mat_frame)
    add_box("back", (hx - t / 2.0, 0.0, height / 2.0), (t, width, height), mat_frame)
    add_box("side_left", (0.0, hy - t / 2.0, height / 2.0), (depth, t, height), mat_frame)
    add_box("side_right", (0.0, -(hy - t / 2.0), height / 2.0), (depth, t, height), mat_frame)
    if top_closed:
        add_box("top", (0.0, 0.0, height - t / 2.0), (depth, width, t), mat_frame)
    fx = -t / 2.0  # centred on the inner cavity (front face at -hx, back panel inner face at hx - t)
    if slot_orientation == "vertical":
        fz = t + inner_h / 2.0
        if vertical_filler_style == "solid":
            # Solid fillers run from each side wall to the slot edge.
            fy = slot_width / 2.0 + filler_width / 2.0
            add_box("filler_left", (fx, fy, fz), (inner_depth, filler_width, inner_h), mat_fill)
            add_box("filler_right", (fx, -fy, fz), (inner_depth, filler_width, inner_h), mat_fill)
        else:
            # Three recognisable folders per side. They are shorter and
            # shallower than the cavity, with small gaps and one leaning folder
            # in each group. The innermost upright folder starts 3 mm beyond
            # the nominal slot edge, preserving the configured opening.
            prop_specs = (
                # y from slot centre, depth, thickness, height, outward lean
                (0.054, 0.235, 0.022, 0.270, 0.0),
                (0.089, 0.250, 0.024, 0.255, 5.0),
                (0.137, 0.225, 0.030, 0.280, 0.0),
            )
            back_inner_x = hx - t
            for side_name, side_sign in (("left", 1.0), ("right", -1.0)):
                for index, (y_offset, prop_depth, prop_thickness, prop_height, lean_deg) in enumerate(prop_specs):
                    lean_x_deg = -side_sign * lean_deg  # top leans away from the slot
                    lean_rad = abs(lean_deg) * math.pi / 180.0
                    rotated_half_height = 0.5 * (
                        prop_height * math.cos(lean_rad) + prop_thickness * math.sin(lean_rad)
                    )
                    center = (
                        back_inner_x - prop_depth / 2.0,
                        side_sign * y_offset,
                        t + rotated_half_height,
                    )
                    add_folder_prop(
                        f"folder_{side_name}_{index}",
                        center,
                        (prop_depth, prop_thickness, prop_height),
                        lean_x_deg,
                        folder_materials[index],
                    )
        slot = {
            "center": [fx, 0.0, fz],
            "half_extents": [inner_depth / 2.0, slot_width / 2.0, inner_h / 2.0],
            "width_axis_local": [0.0, 1.0, 0.0],
            "base_top_z": t,
            "front_x": -hx,
        }
    else:
        # bottom limit: one solid block from the base to the slot floor, full inner
        # width and depth. Its top face is the slot's only limit -- no upper board,
        # so the slot region runs from the floor up to the (open) shelf top and the
        # object is laid flat on the block.
        inner_w = width - 2 * t
        slot_floor = t + slot_bottom_height
        add_box("bottom_limit", (fx, 0.0, t + slot_bottom_height / 2.0), (inner_depth, inner_w, slot_bottom_height), mat_fill)
        slot = {
            "center": [fx, 0.0, (slot_floor + height) / 2.0],
            "half_extents": [inner_depth / 2.0, inner_w / 2.0, (height - slot_floor) / 2.0],
            "width_axis_local": [0.0, 0.0, 1.0],
            "base_top_z": slot_floor,
            "front_x": -hx,
        }
    stage.GetRootLayer().Save()

    manifest = {
        "name": name,
        "usd": usd_path.name,
        "frame": "origin at footprint centre on the bottom face; +x depth (open front is -x), +y width, +z up",
        "params": {
            "inner_depth": inner_depth,
            "slot_width": slot_width,
            "filler_width": filler_width,
            "height": height,
            "panel_thickness": panel_thickness,
            "top_closed": top_closed,
            "slot_orientation": slot_orientation,
            "vertical_filler_style": vertical_filler_style,
            "slot_bottom_height": slot_bottom_height,
            "color": list(color),
            "filler_color": list(filler_color),
        },
        "footprint": {"x_half": hx, "y_half": hy, "height": height},
        "slot": slot,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return usd_path, manifest_path


def ensure_slot_shelf_asset(name: str = "slot_shelf", out_dir: Path | None = None, **params) -> tuple[Path, dict]:
    """Return ``(usd_path, manifest)`` for a slot shelf, generating it if missing or
    if the stored parameters differ from ``params``."""
    out_dir = Path(out_dir) if out_dir is not None else SHELF_ASSET_DIR
    usd_path = out_dir / f"{name}.usda"
    manifest_path = out_dir / f"{name}_manifest.json"
    if usd_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        stored = manifest.get("params", {})

        def _norm(v):
            # JSON round-trips tuples as lists; compare like-for-like.
            return list(v) if isinstance(v, tuple) else v

        if all(k in stored and stored[k] == _norm(v) for k, v in params.items()):
            return usd_path, manifest
    usd_path, manifest_path = build_slot_shelf(out_dir, name=name, **params)
    return usd_path, json.loads(manifest_path.read_text())
